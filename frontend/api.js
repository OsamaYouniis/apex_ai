/**
 * frontend/api.js  — APEX AI Backend Integration Layer (v5)
 * ─────────────────────────────────────────────────────────
 * Injects handleSignIn, handleSignUp, fetchMLPredictions,
 * sendMsg (JWT-aware override), syncProfileToBackend, apexLogout.
 *
 * Add one line before </body> in index.html:
 *   <script src="api.js"></script>
 *
 * Zero changes to HTML/CSS.
 */

const APEX_API = 'http://localhost:8000';

// ── Token storage ──────────────────────────────────────────────────────────────
const Auth = {
  save(token, user) {
    localStorage.setItem('apex-token',   token);
    localStorage.setItem('apex-user-id', String(user.user_id || 1));
    localStorage.setItem('apex-email',   user.email || '');
  },
  token()    { return localStorage.getItem('apex-token') || ''; },
  userId()   { return parseInt(localStorage.getItem('apex-user-id') || '1'); },
  isLoggedIn() { return !!this.token(); },
  clear()    { ['apex-token','apex-user-id','apex-email'].forEach(k => localStorage.removeItem(k)); },
};

// ── HTTP helpers ───────────────────────────────────────────────────────────────
async function apexPost(path, body, auth = false) {
  const h = { 'Content-Type': 'application/json' };
  if (auth && Auth.token()) h['Authorization'] = 'Bearer ' + Auth.token();
  return fetch(APEX_API + path, { method: 'POST', headers: h, body: JSON.stringify(body) });
}
async function apexGet(path, auth = false) {
  const h = {};
  if (auth && Auth.token()) h['Authorization'] = 'Bearer ' + Auth.token();
  return fetch(APEX_API + path, { headers: h });
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function _err(id, msg) {
  let el = document.getElementById(id);
  if (!el) {
    el = document.createElement('div');
    el.id = id;
    el.style.cssText = 'font-size:.82rem;margin-top:.5rem;';
    const ref = document.querySelector('#page-signin .card, #page-signup .card');
    if (ref) ref.appendChild(el);
  }
  el.style.color = '#ff6b35';
  el.textContent = msg;
}
function _clrErr(id) { const e = document.getElementById(id); if (e) e.textContent = ''; }
function _btnState(sel, loading, label) {
  const b = document.querySelector(sel);
  if (!b) return;
  b.disabled = loading;
  b.style.opacity = loading ? '0.65' : '1';
  b.textContent = loading ? '...' : label;
}

// ── Apply server profile to the existing userProfile global ──────────────────
function apexApplyProfile(profile) {
  if (!profile || typeof userProfile === 'undefined') return;
  if (profile.name)           userProfile.name         = profile.name;
  if (profile.age)            userProfile.age          = profile.age;
  if (profile.weight_kg)      userProfile.weight       = profile.weight_kg;
  if (profile.height_cm)      userProfile.height       = profile.height_cm;
  if (profile.activity_level) userProfile.activity     = profile.activity_level;
  if (profile.goal)           userProfile.goal         = profile.goal;
  if (profile.target_weight)  userProfile.targetWeight = profile.target_weight;
  if (profile.gender)         userProfile.gender       = profile.gender;
  localStorage.setItem('apex-profile', JSON.stringify(userProfile));
  document.querySelectorAll('.avatar').forEach(el =>
    el.textContent = (userProfile.name || 'U').charAt(0).toUpperCase());
  if (profile.gender === 'f' && typeof applyTheme === 'function') applyTheme('f');
}

// ── SIGN IN ───────────────────────────────────────────────────────────────────
async function handleSignIn() {
  _clrErr('apex-signin-err');
  const email    = (document.getElementById('signin-email')?.value || '').trim();
  const password = document.getElementById('signin-password')?.value || '';
  if (!email || !password) { _err('apex-signin-err', 'Please enter email and password.'); return; }

  _btnState('#page-signin .btn-primary', true, 'Sign In →');
  try {
    const res  = await apexPost('/auth/signin', { email, password });
    const data = await res.json();
    if (!res.ok) { _err('apex-signin-err', data.detail || 'Sign-in failed.'); return; }

    Auth.save(data.token, data);
    apexApplyProfile(data.profile || {});
    if (typeof showToast    === 'function') showToast('Welcome back, ' + data.name.split(' ')[0] + '! 💪');
    if (typeof updateDashboard === 'function') updateDashboard();
    if (typeof goTo === 'function') goTo('dashboard');
    setTimeout(fetchMLPredictions, 400);
  } catch (e) {
    _err('apex-signin-err', 'Cannot reach server — is the backend running?');
  } finally {
    _btnState('#page-signin .btn-primary', false, 'Sign In →');
  }
}

// ── SIGN UP ───────────────────────────────────────────────────────────────────
async function handleSignUp() {
  _clrErr('apex-signup-err');
  const v = id => document.getElementById(id)?.value || '';
  const name     = v('su-name').trim();
  const email    = v('su-email').trim();
  const password = v('su-password');
  const confirm  = v('su-confirm');
  const age      = parseInt(v('su-age'))    || 25;
  const height   = parseFloat(v('su-height')) || 175;
  const weight   = parseFloat(v('su-weight')) || 70;
  const target   = parseFloat(v('su-target')) || weight - 5;
  const timeframe = v('su-timeframe') || '3-6 months';

  const genderEl = document.getElementById('gender-m');
  const gender   = genderEl?.classList.contains('active') ? 'm' : 'f';

  let goal = 'lose';
  ['lose','build','fit','maintain'].forEach(g => {
    if (document.getElementById('goal-' + g)?.classList.contains('active')) goal = g;
  });

  const actSel = v('su-activity');
  const actMap = { Sedentary: 1, Lightly: 2, Moderately: 3, Very: 4, Extremely: 5 };
  let activity = 2;
  Object.entries(actMap).forEach(([k, n]) => { if (actSel.startsWith(k)) activity = n; });

  if (!name)               { _err('apex-signup-err', 'Please enter your name.'); return; }
  if (!email)              { _err('apex-signup-err', 'Please enter your email.'); return; }
  if (password.length < 6) { _err('apex-signup-err', 'Password must be at least 6 characters.'); return; }
  if (password !== confirm) { _err('apex-signup-err', 'Passwords do not match.'); return; }

  _btnState('#signup-step3 .btn-primary', true, '🚀 Launch APEX AI');
  try {
    const res = await apexPost('/auth/signup', {
      full_name: name, email, password, gender, age,
      weight_kg: weight, height_cm: height, activity_level: activity,
      goal, target_weight: target, dietary_pref: v('su-diet') || 'No Restrictions', timeframe,
    });
    const data = await res.json();
    if (!res.ok) { _err('apex-signup-err', data.detail || 'Sign-up failed.'); return; }

    Auth.save(data.token, data);
    if (typeof userProfile !== 'undefined') {
      Object.assign(userProfile, { name, age, weight, height, activity, goal, targetWeight: target, gender });
      localStorage.setItem('apex-profile', JSON.stringify(userProfile));
    }
    apexApplyProfile(data.profile || {});
    if (typeof showToast    === 'function') showToast('Welcome, ' + name.split(' ')[0] + '! 🚀');
    if (typeof updateDashboard === 'function') updateDashboard();
    if (typeof goTo === 'function') goTo('dashboard');
    setTimeout(fetchMLPredictions, 500);
  } catch (e) {
    _err('apex-signup-err', 'Cannot reach server — is the backend running?');
  } finally {
    _btnState('#signup-step3 .btn-primary', false, '🚀 Launch APEX AI');
  }
}

// ── LOGOUT ────────────────────────────────────────────────────────────────────
function apexLogout() {
  Auth.clear();
  if (typeof showToast === 'function') showToast('Logged out. See you soon! 👋');
  if (typeof goTo     === 'function') goTo('landing');
}

// Patch logout sidebar link
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.sidebar-link').forEach(el => {
    if (el.textContent.includes('Logout')) el.setAttribute('onclick', 'apexLogout()');
  });
});

// ── FETCH ML PREDICTIONS (overrides existing stub) ────────────────────────────
async function fetchMLPredictions() {
  if (typeof userProfile === 'undefined') return;
  try {
    const h = { 'Content-Type': 'application/json' };
    if (Auth.token()) h['Authorization'] = 'Bearer ' + Auth.token();
    const res = await fetch(APEX_API + '/predict', {
      method: 'POST', headers: h,
      body: JSON.stringify({
        age:            userProfile.age      || 25,
        weight_kg:      userProfile.weight   || 70,
        height_cm:      userProfile.height   || 175,
        activity_level: userProfile.activity || 2,
        gender:         userProfile.gender === 'f' ? 0 : 1,
        user_id:        Auth.userId(),
      }),
    });
    if (!res.ok) return;
    const d = await res.json();
    window._mlData = d;

    // Populate all data-ml placeholders
    const set = (sel, val) => document.querySelectorAll(sel).forEach(el => el.textContent = val);
    set('[data-ml="tdee"]',         Math.round(d.calories_tdee) + ' kcal');
    set('[data-ml="fitness-level"]', d.fitness_level);
    set('[data-ml="weight-change"]', (d.weight_change_30d >= 0 ? '+' : '') + d.weight_change_30d.toFixed(1) + ' kg/30d');
    set('[data-ml="bmi"]',           d.bmi);
    set('[data-ml="bmi-cat"]',       d.bmi_category);
    set('[data-ml="protein"]',       d.protein_g + 'g');
    set('[data-ml="water"]',         d.water_l + 'L');
    set('[data-ml="cut-cal"]',       Math.round(d.cut_calories) + ' kcal');
    set('[data-ml="bulk-cal"]',      Math.round(d.bulk_calories) + ' kcal');

    // Calorie KPI card
    const kpi = document.querySelector('[data-kpi="calories"]');
    if (kpi) kpi.textContent = Math.round(d.calories_tdee) + ' kcal';

    // Fitness level badge
    const badge = document.getElementById('fitness-level-badge');
    if (badge) {
      badge.textContent = d.fitness_level;
      badge.style.color = { Beginner:'#00ff88', Intermediate:'#00d4ff', Advanced:'#7b5cff' }[d.fitness_level] || 'var(--neon)';
    }

    // Recommendations list
    const recsEl = document.getElementById('ai-recommendations');
    if (recsEl && d.recommendations?.length) {
      recsEl.innerHTML = d.recommendations.slice(0, 5).map(r =>
        '<div style="display:flex;align-items:center;gap:.75rem;padding:.6rem 0;border-bottom:1px solid var(--border);">' +
        '<span style="width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,var(--neon),var(--neon2));display:flex;align-items:center;justify-content:center;font-size:.7rem;flex-shrink:0;color:var(--dark);font-weight:700;">AI</span>' +
        '<div><div style="font-size:.9rem;font-weight:600;">' + (r.exercise || r) + '</div>' +
        '<div style="font-size:.75rem;color:var(--text3);">' + (r.avg_sets||3) + ' sets · ' + (r.avg_reps||10) + ' reps</div></div></div>'
      ).join('');
    }
    console.log('ML predictions loaded:', d.fitness_level, d.calories_tdee + ' kcal');
    return d;
  } catch (e) {
    console.warn('Backend offline — offline mode active.', e.message);
  }
}

// ── CHAT OVERRIDE (adds JWT + richer user profile) ────────────────────────────
window.sendMsg = async function() {
  const input = document.getElementById('chat-input-field');
  const text  = input?.value.trim();
  if (!text) return;
  if (input) input.value = '';

  const container = document.getElementById('chat-messages');
  const initials  = (userProfile?.name || 'U').charAt(0).toUpperCase();

  const userDiv = document.createElement('div');
  userDiv.className = 'chat-msg user';
  userDiv.innerHTML = '<div style="width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#ff6b35,#ffd93d);display:flex;align-items:center;justify-content:center;font-size:.85rem;flex-shrink:0;font-weight:700;">' + initials + '</div><div><div class="chat-bubble">' + text.replace(/</g,'&lt;') + '</div><div style="font-size:.68rem;color:var(--text3);margin-top:.3rem;text-align:right;padding-right:.5rem;">You · Just now</div></div>';
  container?.appendChild(userDiv);

  const typingDiv = document.createElement('div');
  typingDiv.className = 'chat-msg ai'; typingDiv.id = 'typing-indicator';
  typingDiv.innerHTML = '<div style="width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,var(--neon),var(--neon2));display:flex;align-items:center;justify-content:center;font-size:.9rem;flex-shrink:0;">🧠</div><div class="chat-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>';
  container?.appendChild(typingDiv);
  if (container) container.scrollTop = container.scrollHeight;

  const hist = typeof chatHistory !== 'undefined' ? chatHistory : [];
  hist.push({ role: 'user', content: text });
  if (hist.length > 20) hist.shift();

  const ud = userProfile || {};
  const levelMap = {
    1: 'sedentary',
    2: 'light',
    3: 'moderate',
    4: 'active',
    5: 'very_active',
  };
  const normalizedActivity = levelMap[Number(ud.activity)] || 'moderate';
  const normalizedGender = ud.gender === 'f' ? 'female' : 'male';
  let aiText = '';
  try {
    const h = { 'Content-Type': 'application/json' };
    if (Auth.token()) h['Authorization'] = 'Bearer ' + Auth.token();
    const response = await fetch(APEX_API + '/rag/chat', {
      method: 'POST', headers: h,
      body: JSON.stringify({
        message: text,
        user_id: String(Auth.userId()),
        age: ud.age || 25,
        weight: ud.weight || 70,
        height: ud.height || 175,
        gender: normalizedGender,
        goal: ud.goal || 'cut',
        activity_level: normalizedActivity,
      }),
    });
    document.getElementById('typing-indicator')?.remove();
    if (response.ok) {
      const data = await response.json();
      aiText = data.response || data.reply || 'Let me help! 💪';
      console.log('RAG chat calories:', data.calories, '| macros:', data.macros);
    } else {
      aiText = typeof getFallbackResponse === 'function' ? getFallbackResponse(text) : 'How can I help with your fitness today? 💪';
    }
  } catch (_) {
    document.getElementById('typing-indicator')?.remove();
    aiText = typeof getFallbackResponse === 'function' ? getFallbackResponse(text) : 'Backend offline — start the server on port 8000. 🔧';
  }

  hist.push({ role: 'assistant', content: aiText });
  const aiDiv = document.createElement('div');
  aiDiv.className = 'chat-msg ai';
  const fmt = aiText.replace(/</g,'&lt;').replace(/\n/g,'<br>').replace(/\*\*(.*?)\*\*/g,'<b>$1</b>');
  aiDiv.innerHTML = '<div style="width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,var(--neon),var(--neon2));display:flex;align-items:center;justify-content:center;font-size:.9rem;flex-shrink:0;">🧠</div><div><div class="chat-bubble">' + fmt + '</div><div style="font-size:.68rem;color:var(--text3);margin-top:.3rem;padding-left:.5rem;">APEX AI · Just now</div></div>';
  container?.appendChild(aiDiv);
  if (container) container.scrollTop = container.scrollHeight;
};

// ── SYNC PROFILE (JWT-aware override) ─────────────────────────────────────────
window.syncProfileToBackend = async function() {
  if (!Auth.isLoggedIn() || typeof userProfile === 'undefined') return;
  const h = { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + Auth.token() };
  try {
    await fetch(APEX_API + '/user-data/profile', {
      method: 'POST', headers: h,
      body: JSON.stringify({
        name: userProfile.name, age: userProfile.age,
        weight_kg: userProfile.weight, height_cm: userProfile.height,
        activity_level: userProfile.activity, gender: userProfile.gender === 'f' ? 0 : 1,
        goal: userProfile.goal, target_weight: userProfile.targetWeight, user_id: Auth.userId(),
      }),
    });
  } catch (_) {}
};

// ── SESSION RESTORE ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  if (!Auth.isLoggedIn()) return;
  apexGet('/auth/me', true)
    .then(r => r.ok ? r.json() : null)
    .then(d => { if (d?.authenticated) setTimeout(fetchMLPredictions, 800); else Auth.clear(); })
    .catch(() => {});
});

// ── DASHBOARD REFRESH ON NAVIGATE ─────────────────────────────────────────────
const _origGoTo = window.goTo;
if (typeof _origGoTo === 'function') {
  window.goTo = function(page) { _origGoTo(page); if (page === 'dashboard') setTimeout(fetchMLPredictions, 200); };
}

// ── STARTUP HEALTH CHECK ──────────────────────────────────────────────────────
window.addEventListener('load', () => {
  fetch(APEX_API + '/health')
    .then(r => r.json())
    .then(d => { console.log('APEX AI backend:', d.status, d.models); fetchMLPredictions(); })
    .catch(() => console.warn('Backend offline — client-only mode'));
});
