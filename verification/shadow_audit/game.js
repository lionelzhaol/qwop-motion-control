const GameEngine = (() => {
  const STATIC_BASE = (window.RUNNING_STATIC_BASE || '').replace(/\/+$/, '');
  const MODE_MAP = {
    sprint:      { arg: 'classic',   badge: '100M',   ruleset: 'classic_100m', index: 0 },
    hurdle:      { arg: 'hurdle',    badge: '110M H', ruleset: 'hurdle_110m', index: 1 },
    'long-jump': { arg: 'long_jump', badge: 'LJ',     ruleset: 'long_jump_30m', index: 2 },
  };

  const RULESET_TO_MODE = {};
  for (var k in MODE_MAP) RULESET_TO_MODE[MODE_MAP[k].ruleset] = k;

  const INDEX_TO_MODE = ['sprint', 'hurdle', 'long-jump'];
  const REPLAY_RULESET_TO_MODE = { 1: 'sprint', 2: 'hurdle', 3: 'long-jump' };

  let currentMode = null;
  let wasmReady = false;
  let lastReplay = null;
  let ghostRequestId = 0;

  const canvas = document.getElementById('game-canvas');
  const overlay = document.getElementById('game-overlay');
  const loadText = document.getElementById('load-text');
  const loadProgress = document.getElementById('load-progress');

  function showOverlay(visible) {
    overlay.classList.toggle('is-hidden', !visible);
  }

  function setProgress(ratio) {
    loadProgress.style.width = (ratio * 100) + '%';
  }

  function boot(mode) {
    var config = MODE_MAP[mode];
    if (!config) return;
    currentMode = mode;
    showOverlay(true);
    loadText.textContent = '正在加载游戏…';
    setProgress(0);

    window.Module = {
      canvas: canvas,
      arguments: ['--game', config.arg],
      locateFile: function (path) { return STATIC_BASE + '/game/' + path; },
      setStatus: function (text) {
        if (!text) {
          showOverlay(false);
          canvas.focus();
          wasmReady = true;
          if (typeof window.__onWasmReady === 'function') window.__onWasmReady();
          return;
        }
        var match = text.match(/([^(]+)\((\d+)\/(\d+)\)/);
        if (match) {
          var total = parseInt(match[3]);
          var done = parseInt(match[2]);
          setProgress(total > 0 ? (total - done) / total : 0);
          loadText.textContent = match[1].trim() || '正在加载…';
        } else {
          loadText.textContent = text;
        }
      },
      monitorRunDependencies: function () {},
      print: function (text) { console.log('[game]', text); },
      printErr: function (text) { console.warn('[game]', text); },
    };

    var script = document.createElement('script');
    script.src = STATIC_BASE + '/game/running_ai.js';
    script.async = true;
    script.onerror = function () {
      loadText.textContent = '加载失败，请刷新页面重试';
    };
    document.body.appendChild(script);
  }

  function switchMode(mode) {
    var config = MODE_MAP[mode];
    if (!config) return;
    if (!wasmReady) return;
    currentMode = mode;
    lastReplay = null;
    Module._js_switch_game_mode(config.index);
  }

  function syncMode(mode) {
    if (!MODE_MAP[mode]) return false;
    currentMode = mode;
    lastReplay = null;
    return true;
  }

  function loadGhost(ruleset) {
    if (!wasmReady) return Promise.resolve(false);
    var requestId = ++ghostRequestId;
    Module._js_clear_ghost_replay();
    return fetch('/api/rank/' + ruleset + '/top-replay').then(function (resp) {
      if (requestId !== ghostRequestId) return null;
      if (!resp.ok) return new ArrayBuffer(0);
      return resp.arrayBuffer();
    }).then(function (buf) {
      if (buf === null || requestId !== ghostRequestId) return false;
      if (buf.byteLength > 0) {
        var bytes = new Uint8Array(buf);
        var ptr = Module._malloc(bytes.length);
        Module.HEAPU8.set(bytes, ptr);
        Module._js_set_ghost_replay(ptr, bytes.length);
        Module._free(ptr);
      }
      return true;
    }).catch(function () {
      if (requestId === ghostRequestId) {
        Module._js_clear_ghost_replay();
        return true;
      }
      return false;
    });
  }

  function clearGhost() {
    if (!wasmReady) return;
    ghostRequestId++;
    Module._js_clear_ghost_replay();
  }

  function playReplay(bytes) {
    if (!wasmReady) return false;
    if (!modeFromReplay(bytes)) return false;
    var ptr = Module._malloc(bytes.length);
    Module.HEAPU8.set(bytes, ptr);
    Module._js_play_replay(ptr, bytes.length);
    Module._free(ptr);
    return true;
  }

  function getMode() { return currentMode; }

  function getRuleset() {
    return currentMode ? MODE_MAP[currentMode].ruleset : null;
  }

  function getModeConfig(mode) {
    return MODE_MAP[mode] || null;
  }

  function isReady() { return wasmReady; }

  function getLastReplay() { return lastReplay; }

  function setLastReplay(r) { lastReplay = r; }

  function modeFromIndex(idx) { return INDEX_TO_MODE[idx] || null; }

  function modeFromRuleset(rs) { return RULESET_TO_MODE[rs] || null; }

  function modeFromReplay(bytes) {
    if (!(bytes instanceof Uint8Array) || bytes.length < 14) return null;
    if (bytes[0] !== 0x52 || bytes[1] !== 0x55 || bytes[2] !== 0x4E || bytes[3] !== 0x52) return null;
    return REPLAY_RULESET_TO_MODE[bytes[5]] || null;
  }

  return {
    boot, switchMode, syncMode, loadGhost, clearGhost, playReplay,
    getMode, getRuleset, getModeConfig, isReady,
    getLastReplay, setLastReplay, modeFromIndex, modeFromRuleset, modeFromReplay,
    MODE_MAP,
  };
})();
