var pc = null;
var remoteStream = null;
var retryWithoutStunTried = false;
var connectWatchdogTimer = null;
var disconnectedFailTimer = null;
var manualStopRequested = false;
var AUTO_RETRY_WITHOUT_STUN = false;
var startAttemptSerial = 0;
var iceConfigCache = {};

var DEFAULT_ICE_SERVERS = [
    { urls: ['stun:stun.l.google.com:19302', 'stun:stun1.l.google.com:19302', 'stun:stun.freeswitch.org:3478'] }
];

var ICE_GATHER_TIMEOUT_MS = 3000;
var CONNECT_TIMEOUT_MS = 20000;
var DISCONNECTED_GRACE_MS = 5000;

function isLoopbackHostname() {
    var host = (window.location && window.location.hostname) || '';
    return host === '127.0.0.1' || host === 'localhost' || host === '::1' || host === '[::1]';
}

function shouldUseIceServers(defaultValue) {
    var useStunEl = document.getElementById('use-stun');
    if (!useStunEl) {
        return !!defaultValue;
    }
    return !!useStunEl.checked;
}

function setUseStunCheckbox(enabled) {
    var useStunEl = document.getElementById('use-stun');
    if (useStunEl) {
        useStunEl.checked = !!enabled;
    }
}

function applyDefaultIceServerPreference() {
    var useStunEl = document.getElementById('use-stun');
    if (!useStunEl) {
        return;
    }
    if (isLoopbackHostname()) {
        useStunEl.checked = false;
    }
}

function cloneIceServers(iceServers) {
    return (iceServers || []).map(function(entry) {
        var cloned = {};
        Object.keys(entry || {}).forEach(function(key) {
            if (key === 'urls' && Array.isArray(entry[key])) {
                cloned[key] = entry[key].slice();
                return;
            }
            cloned[key] = entry[key];
        });
        return cloned;
    });
}

function buildFallbackRtcConfig(useIceServers) {
    var config = {
        sdpSemantics: 'unified-plan'
    };
    if (useIceServers) {
        config.iceServers = cloneIceServers(DEFAULT_ICE_SERVERS);
    }
    return config;
}

function fetchRtcConfig(useIceServers) {
    if (!useIceServers) {
        return Promise.resolve(buildFallbackRtcConfig(false));
    }

    if (iceConfigCache.enabled) {
        return Promise.resolve({
            sdpSemantics: 'unified-plan',
            iceServers: cloneIceServers(iceConfigCache.enabled)
        });
    }

    return fetch('/ice-config?enabled=1', { method: 'GET' })
        .then(function(response) {
            if (!response.ok) {
                return response.text().then(function(text) {
                    throw new Error('ICE config failed (' + response.status + '): ' + text);
                });
            }
            return response.json();
        })
        .then(function(payload) {
            if (!payload || !Array.isArray(payload.iceServers)) {
                throw new Error('Invalid ICE config response.');
            }
            iceConfigCache.enabled = cloneIceServers(payload.iceServers);
            return {
                sdpSemantics: 'unified-plan',
                iceServers: cloneIceServers(payload.iceServers)
            };
        })
        .catch(function(err) {
            console.warn('Failed to load /ice-config, falling back to built-in ICE servers.', err);
            return buildFallbackRtcConfig(true);
        });
}

function clearConnectionTimers() {
    if (connectWatchdogTimer) {
        clearTimeout(connectWatchdogTimer);
        connectWatchdogTimer = null;
    }
    if (disconnectedFailTimer) {
        clearTimeout(disconnectedFailTimer);
        disconnectedFailTimer = null;
    }
}

function updateStartStopButtons(isRunning) {
    var startEl = document.getElementById('start');
    var stopEl = document.getElementById('stop');
    if (startEl) {
        startEl.style.display = isRunning ? 'none' : 'inline-block';
    }
    if (stopEl) {
        stopEl.style.display = isRunning ? 'inline-block' : 'none';
    }
}

function notifyConnected() {
    if (typeof window.onWebRTCConnected === 'function') {
        window.onWebRTCConnected();
    }
}

function notifyDisconnected() {
    if (typeof window.onWebRTCDisconnected === 'function') {
        window.onWebRTCDisconnected();
    }
}

function closePeerConnection() {
    if (!pc) {
        return;
    }
    try {
        pc.close();
    } catch (e) {
        console.warn('Error while closing peer connection:', e);
    }
    pc = null;
}

function waitForIceGatheringComplete(activePc, timeoutMs) {
    return new Promise(function(resolve) {
        if (!activePc || activePc.iceGatheringState === 'complete') {
            resolve();
            return;
        }

        var resolved = false;
        var timeoutId = setTimeout(function() {
            if (resolved) {
                return;
            }
            resolved = true;
            activePc.removeEventListener('icegatheringstatechange', onStateChange);
            console.warn('ICE gathering timeout, continue with current SDP.');
            resolve();
        }, timeoutMs);

        function onStateChange() {
            if (resolved) {
                return;
            }
            if (activePc.iceGatheringState === 'complete') {
                resolved = true;
                clearTimeout(timeoutId);
                activePc.removeEventListener('icegatheringstatechange', onStateChange);
                resolve();
            }
        }

        activePc.addEventListener('icegatheringstatechange', onStateChange);
    });
}

function negotiate(activePc, useIceServers) {
    activePc.addTransceiver('video', { direction: 'recvonly' });
    activePc.addTransceiver('audio', { direction: 'recvonly' });

    return activePc.createOffer()
        .then(function(offer) {
            return activePc.setLocalDescription(offer);
        })
        .then(function() {
            return waitForIceGatheringComplete(activePc, ICE_GATHER_TIMEOUT_MS);
        })
        .then(function() {
            if (!activePc.localDescription) {
                throw new Error('LocalDescription is missing.');
            }
            return fetch('/offer', {
                body: JSON.stringify({
                    sdp: activePc.localDescription.sdp,
                    type: activePc.localDescription.type,
                    useIceServers: !!useIceServers,
                }),
                headers: {
                    'Content-Type': 'application/json'
                },
                method: 'POST'
            });
        })
        .then(function(response) {
            if (!response.ok) {
                return response.text().then(function(text) {
                    throw new Error('Offer failed (' + response.status + '): ' + text);
                });
            }
            return response.json();
        })
        .then(function(answer) {
            if (!answer || !answer.sdp || !answer.type) {
                throw new Error('Invalid SDP answer from server.');
            }
            var sidEl = document.getElementById('sessionid');
            if (sidEl && typeof answer.sessionid !== 'undefined') {
                sidEl.value = answer.sessionid;
            }
            if (typeof window.onWebRTCSessionCreated === 'function' && typeof answer.sessionid !== 'undefined') {
                window.onWebRTCSessionCreated(answer.sessionid);
            }
            return activePc.setRemoteDescription({
                sdp: answer.sdp,
                type: answer.type
            });
        });
}

function handleConnectionFailure(activePc, usedIceServers, reason) {
    if (activePc && pc !== activePc) {
        return;
    }
    clearConnectionTimers();

    if (
        AUTO_RETRY_WITHOUT_STUN &&
        !manualStopRequested &&
        usedIceServers &&
        !retryWithoutStunTried
    ) {
        console.warn('WebRTC failed with ICE servers, retry with host candidates only. Reason:', reason);
        retryWithoutStunTried = true;
        setUseStunCheckbox(false);
        closePeerConnection();
        runStartAttempt({ useIceServers: false, isRetry: true });
        return;
    }

    if (!manualStopRequested) {
        console.error('WebRTC connection failed:', reason);
        if (reason) {
            alert(typeof reason === 'string' ? reason : (reason.message || String(reason)));
        }
    }

    closePeerConnection();
    updateStartStopButtons(false);
    notifyDisconnected();
}

function runStartAttempt(options) {
    options = options || {};
    manualStopRequested = false;
    var usedIceServers = typeof options.useIceServers === 'boolean' ? options.useIceServers : shouldUseIceServers(false);
    var attemptId = ++startAttemptSerial;

    clearConnectionTimers();
    closePeerConnection();
    updateStartStopButtons(true);
    fetchRtcConfig(usedIceServers)
        .then(function(config) {
            if (manualStopRequested || attemptId !== startAttemptSerial) {
                return;
            }

            pc = new RTCPeerConnection(config);
            var activePc = pc;

            activePc.addEventListener('iceconnectionstatechange', function() {
                if (pc !== activePc) {
                    return;
                }
                console.log('ICE state:', activePc.iceConnectionState);
            });

            activePc.addEventListener('icegatheringstatechange', function() {
                if (pc !== activePc) {
                    return;
                }
                console.log('ICE gathering:', activePc.iceGatheringState);
            });

            activePc.addEventListener('icecandidateerror', function(evt) {
                if (pc !== activePc) {
                    return;
                }
                console.warn('ICE candidate error:', evt);
            });

            activePc.addEventListener('connectionstatechange', function() {
                if (pc !== activePc) {
                    return;
                }
                console.log('PC state:', activePc.connectionState);

                if (activePc.connectionState === 'connected') {
                    clearConnectionTimers();
                    notifyConnected();
                    return;
                }

                if (activePc.connectionState === 'disconnected') {
                    if (disconnectedFailTimer) {
                        clearTimeout(disconnectedFailTimer);
                    }
                    disconnectedFailTimer = setTimeout(function() {
                        if (pc === activePc && activePc.connectionState === 'disconnected') {
                            handleConnectionFailure(activePc, usedIceServers, 'Connection stayed disconnected');
                        }
                    }, DISCONNECTED_GRACE_MS);
                    return;
                }

                if (activePc.connectionState === 'failed' || activePc.connectionState === 'closed') {
                    handleConnectionFailure(activePc, usedIceServers, 'Connection state: ' + activePc.connectionState);
                }
            });

            remoteStream = new MediaStream();
            activePc.addEventListener('track', function(evt) {
                if (pc !== activePc) {
                    return;
                }
                var videoEl = document.getElementById('video');
                var audioEl = document.getElementById('audio');

                remoteStream.addTrack(evt.track);

                if (videoEl) {
                    videoEl.srcObject = remoteStream;
                    videoEl.muted = false;
                    videoEl.play().catch(function() {});
                }
                if (audioEl) {
                    audioEl.srcObject = remoteStream;
                    audioEl.muted = false;
                    audioEl.play().catch(function() {});
                }
            });

            connectWatchdogTimer = setTimeout(function() {
                if (pc !== activePc) {
                    return;
                }
                if (activePc.connectionState !== 'connected') {
                    handleConnectionFailure(activePc, usedIceServers, 'Connect timeout after ' + CONNECT_TIMEOUT_MS + 'ms');
                }
            }, CONNECT_TIMEOUT_MS);

            negotiate(activePc, usedIceServers).catch(function(err) {
                handleConnectionFailure(activePc, usedIceServers, err);
            });
        })
        .catch(function(err) {
            handleConnectionFailure(null, usedIceServers, err);
        });
}

function start(options) {
    options = options || {};
    if (!options.isRetry) {
        retryWithoutStunTried = false;
    }
    runStartAttempt(options);
}

function stop() {
    manualStopRequested = true;
    startAttemptSerial += 1;
    retryWithoutStunTried = false;
    clearConnectionTimers();
    updateStartStopButtons(false);

    var sidEl = document.getElementById('sessionid');
    if (sidEl) {
        sidEl.value = '0';
    }

    closePeerConnection();
    notifyDisconnected();
}

window.onunload = function() {
    manualStopRequested = true;
    startAttemptSerial += 1;
    clearConnectionTimers();
    closePeerConnection();
};

window.onbeforeunload = function(e) {
    manualStopRequested = true;
    startAttemptSerial += 1;
    clearConnectionTimers();
    closePeerConnection();
    e = e || window.event;
    if (e) {
        e.returnValue = '关闭提示';
    }
    return '关闭提示';
};

applyDefaultIceServerPreference();
