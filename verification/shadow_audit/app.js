(function () {
  var STATIC_BASE = (window.RUNNING_STATIC_BASE || '').replace(/\/+$/, '');
  var DEFAULT_MODE = 'sprint';
  var TAUNTS = [
    { img: STATIC_BASE + '/assets/rotifer.jpg', alt: '轮虫嘲讽卡片' },
    { img: STATIC_BASE + '/assets/c_elegans.jpg', alt: '秀丽隐杆线虫嘲讽卡片' },
    { img: STATIC_BASE + '/assets/hydra.jpg', alt: '水螅嘲讽卡片' },
    { img: STATIC_BASE + '/assets/sea_squirt.jpg', alt: '玻璃海鞘嘲讽卡片' },
  ];
  var WIN = { img: STATIC_BASE + '/assets/win.jpg', alt: '玩家追平或超过排行榜第一名' };

  var tauntCard = document.getElementById('taunt-card');
  var tauntImg = document.getElementById('taunt-img');
  var uploadedReplaySequence = 0;
  var finishedRunSequence = 0;

  function hideTaunt() {
    tauntCard.classList.add('is-hidden');
  }

  function showCard(card) {
    tauntImg.src = card.img;
    tauntImg.alt = card.alt;
    tauntImg.classList.remove('is-hidden');
    tauntCard.classList.remove('is-hidden');
    tauntCard.focus();
  }

  function showTaunt() {
    showCard(TAUNTS[Math.floor(Math.random() * TAUNTS.length)]);
  }

  function showWin() {
    showCard(WIN);
  }

  function getInitialMode() {
    var params = new URLSearchParams(window.location.search);
    var mode = params.get('mode');
    if (mode && GameEngine.MODE_MAP[mode]) return mode;
    return DEFAULT_MODE;
  }

  function loadGhostIfEnabled() {
    var ruleset = GameEngine.getRuleset();
    if (ruleset && UI.isGhostEnabled()) {
      GameEngine.loadGhost(ruleset);
    } else {
      GameEngine.clearGhost();
    }
  }

  function switchMode(mode) {
    uploadedReplaySequence++;
    finishedRunSequence++;
    hideTaunt();
    if (GameEngine.isReady()) {
      GameEngine.switchMode(mode);
    }
    GameEngine.setLastReplay(null);
    Leaderboard.enableSubmit(false);
    Leaderboard.startPolling(mode);
    loadGhostIfEnabled();
  }

  window.__onRunFinished = function (replayBytes, success, score, time, fromUpload, shadowFinished, shadowActive) {
    var sequence = ++finishedRunSequence;
    var mode = GameEngine.getMode();
    var config = GameEngine.getModeConfig(mode);
    if (!config) return;

    if (config.ruleset === 'long_jump_30m') {
      UI.setResult((score).toFixed(3) + ' m');
    } else {
      UI.setResult((time).toFixed(3) + ' s');
    }

    if (success && replayBytes) {
      GameEngine.setLastReplay(replayBytes);
      Leaderboard.enableSubmit(true);
    } else {
      GameEngine.setLastReplay(null);
      Leaderboard.enableSubmit(false);
    }

    if (success) {
      var metric = config.ruleset === 'long_jump_30m'
        ? Math.round(score * 1000)
        : Math.round(time * 30);
      Leaderboard.refresh().then(function (refreshed) {
        if (sequence !== finishedRunSequence || GameEngine.getMode() !== mode) return;
        if (refreshed && Leaderboard.isTopResult(mode, metric)) {
          showWin();
        } else if (shadowFinished) {
          showTaunt();
        }
      });
    } else if (shadowFinished || (!success && shadowActive)) {
      showTaunt();
    }
  };

  window.__onModeChanged = function (modeIndex) {
    var mode = GameEngine.modeFromIndex(modeIndex);
    if (mode && mode !== GameEngine.getMode()) {
      finishedRunSequence++;
      GameEngine.syncMode(mode);
      UI.syncModeButtons(mode);
      Leaderboard.startPolling(mode);
      loadGhostIfEnabled();
    }
  };

  window.__onWasmReady = function () {
    loadGhostIfEnabled();
  };

  window.__onRunRestarted = function () {
    finishedRunSequence++;
    hideTaunt();
    GameEngine.setLastReplay(null);
    Leaderboard.enableSubmit(false);
    UI.setResult('--.---');
  };

  UI.initPlayerName();
  UI.initModeSwitcher(switchMode);
  UI.initTouchControls();
  UI.initToolbar();
  UI.initInfoDialog();
  tauntCard.addEventListener('click', function () {
    hideTaunt();
    document.getElementById('game-canvas').focus();
  });
  tauntCard.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' || event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      hideTaunt();
      document.getElementById('game-canvas').focus();
    }
  });

  UI.initGhostToggle(function (enabled) {
    uploadedReplaySequence++;
    if (enabled) {
      loadGhostIfEnabled();
    } else {
      GameEngine.clearGhost();
    }
  });

  UI.initUpload(function (bytes) {
    if (bytes.length > 15014) {
      UI.toast('文件过大，最多支持 30000 帧', true);
      return;
    }
    var mode = GameEngine.modeFromReplay(bytes);
    if (!mode) {
      UI.toast('无效的 replay 文件', true);
      return;
    }

    var sequence = ++uploadedReplaySequence;
    finishedRunSequence++;
    var config = GameEngine.getModeConfig(mode);
    hideTaunt();
    GameEngine.syncMode(mode);
    UI.syncModeButtons(mode);
    GameEngine.setLastReplay(null);
    Leaderboard.enableSubmit(false);
    Leaderboard.startPolling(mode);

    var ghostReady;
    if (UI.isGhostEnabled()) {
      ghostReady = GameEngine.loadGhost(config.ruleset);
    } else {
      GameEngine.clearGhost();
      ghostReady = Promise.resolve(true);
    }

    ghostReady.then(function (ready) {
      if (!ready || sequence !== uploadedReplaySequence || GameEngine.getMode() !== mode) return;
      if (!GameEngine.playReplay(bytes)) {
        UI.toast('无效的 replay 文件', true);
      }
    });
  });

  document.getElementById('submit-btn').addEventListener('click', function () {
    var name = UI.getPlayerName();
    if (!name || !name.trim()) {
      UI.toast('请输入玩家名', true);
      return;
    }
    var replay = GameEngine.getLastReplay();
    if (!replay) {
      UI.toast('没有可提交的成绩', true);
      return;
    }
    Leaderboard.submitReplay(name, replay).then(function (data) {
      if (data && data.qualified) {
        Leaderboard.refresh();
      }
    });
  });

  var initialMode = getInitialMode();
  GameEngine.boot(initialMode);
  Leaderboard.startPolling(initialMode);
  UI.syncModeButtons(initialMode);
})();
