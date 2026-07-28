/**
 * MindSense AI — Chat Interface JavaScript
 * ==========================================
 * Handles:
 *   - Message sending (POST /api/chat)
 *   - Markdown rendering (marked.js + DOMPurify)
 *   - Typing animation & streaming simulation
 *   - Sidebar stats update (condition, risk, intent, quality)
 *   - Crisis modal auto-trigger
 *   - Session reset
 *   - Dark/light theme toggle
 *   - Auto-growing textarea
 *   - Character counter
 *   - Toast notifications
 *   - System status polling
 */

'use strict';

/* ──────────────────────────────────────────────────────────────
   CONFIGURATION
────────────────────────────────────────────────────────────── */
const CONFIG = {
  API_CHAT:    '/api/chat',
  API_RESET:   '/api/reset',
  API_STATUS:  '/api/status',
  MAX_CHARS:   2000,
  STREAM_DELAY: 12,   // ms per character in simulated stream
  STREAM_CHUNK: 3,    // chars per stream frame
};

/* ──────────────────────────────────────────────────────────────
   DOM REFERENCES
────────────────────────────────────────────────────────────── */
const dom = {
  feed:           document.getElementById('messageFeed'),
  form:           document.getElementById('chatForm'),
  input:          document.getElementById('userInput'),
  sendBtn:        document.getElementById('sendBtn'),
  sendIcon:       document.getElementById('sendIcon'),
  charCounter:    document.getElementById('charCounter'),
  typingIndicator:document.getElementById('typingIndicator'),
  welcomeScreen:  document.getElementById('welcomeScreen'),
  starterChips:   document.getElementById('starterChips'),
  statusDot:      document.getElementById('statusDot'),
  statusText:     document.getElementById('statusText'),
  themeToggleBtn: document.getElementById('themeToggleBtn'),
  themeIcon:      document.getElementById('themeIcon'),
  clearChatBtn:   document.getElementById('clearChatBtn'),
  msgCount:       document.getElementById('msgCount'),
  conditionBadge: document.getElementById('conditionBadge'),
  riskBadge:      document.getElementById('riskBadge'),
  intentBadge:    document.getElementById('intentBadge'),
  qualityBadge:   document.getElementById('qualityBadge'),
  sourcesPanel:   document.getElementById('sourcesPanel'),
  sourcesList:    document.getElementById('sourcesList'),
  sourcesCloseBtn:document.getElementById('sourcesCloseBtn'),
  toastContainer: document.getElementById('toastContainer'),
};

/* ──────────────────────────────────────────────────────────────
   STATE
────────────────────────────────────────────────────────────── */
let state = {
  sessionId:   null,
  messageCount: 0,
  isLoading:   false,
  theme:       'dark',
};

/* ──────────────────────────────────────────────────────────────
   MARKED.JS CONFIGURATION
────────────────────────────────────────────────────────────── */
if (typeof marked !== 'undefined') {
  marked.setOptions({
    breaks: true,
    gfm: true,
    headerIds: false,
    mangle: false,
  });
}

function renderMarkdown(text) {
  if (typeof marked === 'undefined') return escapeHtml(text);
  try {
    const html = marked.parse(text || '');
    return typeof DOMPurify !== 'undefined'
      ? DOMPurify.sanitize(html)
      : html;
  } catch (e) {
    return escapeHtml(text);
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/* ──────────────────────────────────────────────────────────────
   THEME MANAGEMENT
────────────────────────────────────────────────────────────── */
function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('mindsense-theme', theme);
  dom.themeIcon.className = theme === 'dark'
    ? 'bi bi-sun-fill'
    : 'bi bi-moon-stars-fill';
  dom.themeToggleBtn.title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
}

function toggleTheme() {
  applyTheme(state.theme === 'dark' ? 'light' : 'dark');
}

/* ──────────────────────────────────────────────────────────────
   STATUS INDICATOR
────────────────────────────────────────────────────────────── */
function setStatus(status, text) {
  dom.statusDot.className = 'status-dot ' + status;
  dom.statusText.textContent = text;
}

async function checkSystemStatus() {
  try {
    const res = await fetch(CONFIG.API_STATUS);
    if (res.ok) {
      const data = await res.json();
      const faissReady = data?.components?.faiss_index?.ready;
      const geminiOk   = data?.components?.gemini_api?.configured;
      if (geminiOk) {
        setStatus('online', faissReady ? 'Online · RAG Active' : 'Online · No Index');
      } else {
        setStatus('offline', 'API Key Missing');
      }
    } else {
      setStatus('offline', 'Server Error');
    }
  } catch {
    setStatus('offline', 'Disconnected');
  }
}

/* ──────────────────────────────────────────────────────────────
   TOAST NOTIFICATIONS
────────────────────────────────────────────────────────────── */
function showToast(message, type = 'success', duration = 3500) {
  const icons = { success: 'bi-check-circle-fill', error: 'bi-x-circle-fill', warning: 'bi-exclamation-triangle-fill' };
  const toast = document.createElement('div');
  toast.className = `ms-toast ${type}`;
  toast.innerHTML = `<i class="bi ${icons[type] || icons.success}"></i><span>${escapeHtml(message)}</span>`;
  dom.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = 'opacity 0.3s, transform 0.3s';
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(110%)';
    setTimeout(() => toast.remove(), 310);
  }, duration);
}

/* ──────────────────────────────────────────────────────────────
   SIDEBAR STATS
────────────────────────────────────────────────────────────── */
function updateSidebar(data) {
  state.messageCount++;
  dom.msgCount.textContent = state.messageCount;

  const label = data?.classification?.label || '—';
  const conf = data?.classification?.confidence ? ` (${(data.classification.confidence * 100).toFixed(0)}%)` : '';
  dom.conditionBadge.textContent = label + conf;

  // LIME Token Attribution Highlights
  const words = data?.classification?.explainable_words || [];
  if (words.length > 0) {
    const wordChips = words.map(w => `<span class="badge bg-secondary me-1" style="font-size:0.65rem">${w.word}</span>`).join('');
    let limeContainer = document.getElementById('limeChipsContainer');
    if (!limeContainer) {
      limeContainer = document.createElement('div');
      limeContainer.id = 'limeChipsContainer';
      limeContainer.className = 'mt-1 text-wrap';
      dom.conditionBadge.parentNode.appendChild(limeContainer);
    }
    limeContainer.innerHTML = `<small class="text-muted d-block mb-1" style="font-size:0.65rem">Key terms (LIME):</small>${wordChips}`;
  }

  const risk = (data?.risk_level || 'low').toLowerCase();
  dom.riskBadge.textContent = risk;
  dom.riskBadge.className = `stat-value risk-badge ${risk}`;

  const intent = (data?.intent || 'general').replace(/_/g, ' ');
  dom.intentBadge.textContent = intent;

  const score = data?.quality_score ?? data?.validation?.score;
  if (score !== undefined && score !== null) {
    dom.qualityBadge.textContent = (score * 100).toFixed(0) + '%';
  }
}

/* ──────────────────────────────────────────────────────────────
   SOURCES PANEL
────────────────────────────────────────────────────────────── */
function showSources(sources) {
  if (!sources || sources.length === 0) {
    dom.sourcesPanel.hidden = true;
    return;
  }
  dom.sourcesList.innerHTML = sources.map(src =>
    `<div class="source-tag"><i class="bi bi-file-earmark-text"></i>${escapeHtml(src)}</div>`
  ).join('');
  dom.sourcesPanel.hidden = false;
}

/* ──────────────────────────────────────────────────────────────
   CRISIS MODAL
────────────────────────────────────────────────────────────── */
function triggerCrisisModal() {
  const modal = document.getElementById('crisisModal');
  if (modal && typeof bootstrap !== 'undefined') {
    const bsModal = new bootstrap.Modal(modal, { backdrop: 'static' });
    bsModal.show();
  }
}

/* ──────────────────────────────────────────────────────────────
   MESSAGE RENDERING
────────────────────────────────────────────────────────────── */
function createTimestamp() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function appendUserMessage(text) {
  hideWelcomeScreen();

  const row = document.createElement('div');
  row.className = 'message-row user-row';
  row.innerHTML = `
    <div class="msg-avatar"><i class="bi bi-person-fill"></i></div>
    <div>
      <div class="msg-bubble user-bubble">
        <div class="msg-content">${escapeHtml(text)}</div>
      </div>
      <div class="msg-meta">
        <span>${createTimestamp()}</span>
        <i class="bi bi-check2-all" style="color:rgba(255,255,255,0.6)"></i>
      </div>
    </div>`;
  dom.feed.appendChild(row);
  scrollToBottom();
  return row;
}

function appendAIMessage(text, sources = []) {
  const row = document.createElement('div');
  row.className = 'message-row ai-row';

  const hasSourcesBtn = sources.length > 0
    ? `<button class="sources-btn" onclick="showSources(${JSON.stringify(sources).replace(/"/g, '&quot;')})">
         <i class="bi bi-journal-bookmark"></i> ${sources.length} source${sources.length > 1 ? 's' : ''}
       </button>`
    : '';

  row.innerHTML = `
    <div class="msg-avatar"><i class="bi bi-heart-pulse-fill"></i></div>
    <div style="max-width:68%">
      <div class="msg-bubble ai-bubble">
        <div class="msg-content" id="aiContent_${Date.now()}">${renderMarkdown(text)}</div>
        ${hasSourcesBtn}
      </div>
      <div class="msg-meta">
        <span>MindSense AI</span>
        <span>·</span>
        <span>${createTimestamp()}</span>
      </div>
    </div>`;
  dom.feed.appendChild(row);
  scrollToBottom();
  return row;
}

/**
 * Append an AI message that fills in character by character (simulated streaming).
 * Returns the content div for progressive updates.
 */
function appendStreamingMessage() {
  const row = document.createElement('div');
  row.className = 'message-row ai-row';
  const contentId = `stream_${Date.now()}`;

  row.innerHTML = `
    <div class="msg-avatar"><i class="bi bi-heart-pulse-fill"></i></div>
    <div style="max-width:68%">
      <div class="msg-bubble ai-bubble">
        <div class="msg-content" id="${contentId}"><span class="stream-cursor"></span></div>
      </div>
      <div class="msg-meta">
        <span>MindSense AI</span>
        <span>·</span>
        <span>${createTimestamp()}</span>
      </div>
    </div>`;
  dom.feed.appendChild(row);
  scrollToBottom();
  return { row, contentDiv: document.getElementById(contentId) };
}

async function animateText(contentDiv, text) {
  let displayed = '';
  contentDiv.innerHTML = '';

  for (let i = 0; i < text.length; i += CONFIG.STREAM_CHUNK) {
    displayed += text.slice(i, i + CONFIG.STREAM_CHUNK);
    contentDiv.innerHTML = renderMarkdown(displayed) + '<span class="stream-cursor"></span>';
    scrollToBottom();
    await sleep(CONFIG.STREAM_DELAY);
  }

  // Final render without cursor
  contentDiv.innerHTML = renderMarkdown(text);
  scrollToBottom();
}

/* ──────────────────────────────────────────────────────────────
   CORE SEND LOGIC
────────────────────────────────────────────────────────────── */
async function sendMessage(text) {
  if (state.isLoading || !text.trim()) return;

  state.isLoading = true;
  setLoading(true);

  appendUserMessage(text);
  showTypingIndicator(true);

  try {
    const body = { message: text };
    if (state.sessionId) body.session_id = state.sessionId;

    const res = await fetch(CONFIG.API_CHAT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    showTypingIndicator(false);

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `Server error ${res.status}`);
    }

    const data = await res.json();

    // Save session ID
    if (data.session_id) state.sessionId = data.session_id;

    const responseText = data?.response?.message || 'I apologize, I could not generate a response.';
    const sources = data?.sources || [];

    // Animate the response
    const { row, contentDiv } = appendStreamingMessage();

    // Render PyTorch Model Prediction Badge at top of bubble
    const classification = data?.classification;
    if (classification && classification.label) {
      const label = classification.label;
      const confPct = classification.confidence ? `${(classification.confidence * 100).toFixed(0)}%` : '';
      const modelName = classification.model_architecture || 'DistilBERT + BiLSTM + Attention';
      const words = classification.explainable_words || [];
      const wordBadges = words.map(w => `<span class="badge bg-dark border text-light me-1" style="font-size:0.65rem">${w.word}</span>`).join('');

      const predCard = document.createElement('div');
      predCard.className = 'prediction-banner mb-3 p-2 rounded';
      predCard.style.cssText = 'background:rgba(124,111,224,0.12); border:1px solid rgba(124,111,224,0.3); border-radius:10px; font-size:0.8rem;';
      predCard.innerHTML = `
        <div class="d-flex align-items-center justify-content-between flex-wrap gap-1 mb-1">
          <div>
            <i class="bi bi-cpu-fill me-1" style="color:var(--accent-from)"></i>
            <span class="fw-bold">Predicted Category:</span>
            <span class="badge bg-primary ms-1">${escapeHtml(label)} ${confPct}</span>
          </div>
          <small class="text-muted" style="font-size:0.65rem">${escapeHtml(modelName)}</small>
        </div>
        ${words.length > 0 ? `<div class="d-flex align-items-center gap-1 mt-1" style="font-size:0.72rem"><span class="text-secondary"><i class="bi bi-magic me-1"></i>LIME Key Words:</span> ${wordBadges}</div>` : ''}
      `;
      contentDiv.parentNode.insertBefore(predCard, contentDiv);
    }

    await animateText(contentDiv, responseText);

    // Add sources button if sources exist
    if (sources.length > 0) {
      const bubble = row.querySelector('.msg-bubble');
      const btn = document.createElement('button');
      btn.className = 'sources-btn';
      btn.innerHTML = `<i class="bi bi-journal-bookmark"></i> ${sources.length} source${sources.length > 1 ? 's' : ''}`;
      btn.addEventListener('click', () => showSources(sources));
      bubble.appendChild(btn);
    }

    // Update sidebar
    updateSidebar(data);

  } catch (err) {
    showTypingIndicator(false);
    console.error('[MindSense] Error sending message:', err);
    showToast('An error occurred while processing your request.', 'error');
  } finally {
    state.isLoading = false;
    setLoading(false);
    dom.input.focus();
  }
}

/* ──────────────────────────────────────────────────────────────
   UI HELPERS
────────────────────────────────────────────────────────────── */
function updateSidebar(data) {
  if (!data) return;
  state.messageCount++;
  if (dom.msgCount) dom.msgCount.textContent = state.messageCount;

  // PyTorch Classification Badge
  const classification = data.classification;
  if (classification && classification.label && dom.conditionBadge) {
    const label = classification.label;
    const conf = classification.confidence ? ` (${(classification.confidence * 100).toFixed(0)}%)` : '';
    dom.conditionBadge.textContent = `${label}${conf}`;
  }

  // Risk Level
  if (data.risk_level && dom.riskBadge) {
    dom.riskBadge.textContent = data.risk_level.toUpperCase();
    dom.riskBadge.className = `stat-value risk-badge risk-${data.risk_level.toLowerCase()}`;
  }

  // Intent
  if (data.intent && dom.intentBadge) {
    dom.intentBadge.textContent = data.intent.replace('_', ' ');
  }

  // Quality Score
  if (data.quality_score !== undefined && dom.qualityBadge) {
    dom.qualityBadge.textContent = `${(data.quality_score * 100).toFixed(0)}%`;
  }
}
function setLoading(isLoading) {
  dom.sendBtn.disabled = isLoading || dom.input.value.trim().length === 0;
  dom.sendIcon.className = isLoading ? 'bi bi-hourglass-split' : 'bi bi-send-fill';
}

function showTypingIndicator(show) {
  dom.typingIndicator.hidden = !show;
  if (show) scrollToBottom();
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    dom.feed.scrollTop = dom.feed.scrollHeight;
  });
}

function hideWelcomeScreen() {
  if (dom.welcomeScreen && dom.welcomeScreen.parentNode) {
    dom.welcomeScreen.style.opacity = '0';
    dom.welcomeScreen.style.transform = 'translateY(-10px)';
    dom.welcomeScreen.style.transition = 'opacity 0.3s, transform 0.3s';
    setTimeout(() => dom.welcomeScreen.remove(), 310);
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/* ──────────────────────────────────────────────────────────────
   SESSION RESET
────────────────────────────────────────────────────────────── */
async function resetSession() {
  if (state.isLoading) return;

  try {
    const res = await fetch(CONFIG.API_RESET, { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      state.sessionId = data.new_session_id || null;
      state.messageCount = 0;

      // Clear feed and reset sidebar
      dom.feed.innerHTML = '';
      dom.msgCount.textContent = '0';
      dom.conditionBadge.textContent = '—';
      dom.riskBadge.textContent = '—';
      dom.riskBadge.className = 'stat-value risk-badge';
      dom.intentBadge.textContent = '—';
      dom.qualityBadge.textContent = '—';
      dom.sourcesPanel.hidden = true;

      // Re-add welcome screen
      const welcomeHTML = `
        <div class="welcome-screen" id="welcomeScreen">
          <div class="welcome-icon"><i class="bi bi-heart-pulse-fill"></i></div>
          <h1 class="welcome-title">Welcome to MindSense AI</h1>
          <p class="welcome-subtitle">A safe, empathetic space to talk about how you're feeling.<br/>
             Everything you share is private and non-judgmental.</p>
          <div class="starter-chips" id="starterChips">
            <button class="chip" data-msg="I've been feeling really anxious lately and I'm not sure why.">😰 I feel anxious</button>
            <button class="chip" data-msg="I've been struggling with low mood and lack of motivation.">😔 Feeling low</button>
            <button class="chip" data-msg="Work stress is overwhelming me and I can't relax.">😩 Overwhelmed</button>
            <button class="chip" data-msg="I just need someone to talk to right now.">💬 Just need to talk</button>
          </div>
        </div>`;
      dom.feed.innerHTML = welcomeHTML;
      attachChipListeners();

      showToast('New session started.', 'success');
    }
  } catch (err) {
    console.error('[MindSense] Reset error:', err);
    showToast('Could not reset session.', 'error');
  }
}

/* ──────────────────────────────────────────────────────────────
   TEXTAREA AUTO-GROW
────────────────────────────────────────────────────────────── */
function autoGrow(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

/* ──────────────────────────────────────────────────────────────
   EVENT LISTENERS
────────────────────────────────────────────────────────────── */
function attachChipListeners() {
  document.querySelectorAll('.chip[data-msg]').forEach(chip => {
    chip.addEventListener('click', () => {
      const msg = chip.dataset.msg;
      dom.input.value = msg;
      autoGrow(dom.input);
      updateCharCounter(msg.length);
      dom.sendBtn.disabled = false;
      dom.input.focus();
      sendMessage(msg);
      dom.input.value = '';
      autoGrow(dom.input);
      updateCharCounter(0);
    });
  });
}

function updateCharCounter(count) {
  dom.charCounter.textContent = `${count} / ${CONFIG.MAX_CHARS}`;
  dom.charCounter.className = 'char-counter'
    + (count > CONFIG.MAX_CHARS * 0.85 ? ' near-limit' : '')
    + (count >= CONFIG.MAX_CHARS ? ' at-limit' : '');
}

function initEventListeners() {
  // Form submit
  dom.form.addEventListener('submit', e => {
    e.preventDefault();
    const text = dom.input.value.trim();
    if (text && !state.isLoading) {
      dom.input.value = '';
      autoGrow(dom.input);
      updateCharCounter(0);
      dom.sendBtn.disabled = true;
      sendMessage(text);
    }
  });

  // Input events
  dom.input.addEventListener('input', () => {
    autoGrow(dom.input);
    const len = dom.input.value.length;
    updateCharCounter(len);
    dom.sendBtn.disabled = len === 0 || state.isLoading;
  });

  // Enter to send (Shift+Enter for newline)
  dom.input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!dom.sendBtn.disabled) dom.form.dispatchEvent(new Event('submit'));
    }
  });

  // Theme toggle
  dom.themeToggleBtn.addEventListener('click', toggleTheme);

  // Clear chat / session reset
  dom.clearChatBtn.addEventListener('click', () => {
    if (confirm('Start a new conversation? Your current session will be saved.')) {
      resetSession();
    }
  });

  // Sources close
  dom.sourcesCloseBtn.addEventListener('click', () => {
    dom.sourcesPanel.hidden = true;
  });

  // Starter chips
  attachChipListeners();
}

/* ──────────────────────────────────────────────────────────────
   INITIALIZATION
────────────────────────────────────────────────────────────── */
function init() {
  // Restore theme
  const savedTheme = localStorage.getItem('mindsense-theme') || 'dark';
  applyTheme(savedTheme);

  // Initialize event listeners
  initEventListeners();

  // Check system status
  setStatus('loading', 'Connecting…');
  checkSystemStatus();

  // Focus input
  dom.input.focus();

  console.log(
    '%c MindSense AI %c v1.0.0 ',
    'background:#7c6fe0;color:#fff;padding:4px 8px;border-radius:6px 0 0 6px;font-weight:bold',
    'background:#1e2233;color:#8892a4;padding:4px 8px;border-radius:0 6px 6px 0'
  );
}

document.addEventListener('DOMContentLoaded', init);
