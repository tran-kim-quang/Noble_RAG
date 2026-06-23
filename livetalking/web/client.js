var pc = null;

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function postJsonWithRetry(url, payload, options = {}) {
    const retries = Number.isInteger(options.retries) ? options.retries : 2;
    const delayMs = Number.isInteger(options.delayMs) ? options.delayMs : 700;
    const timeoutMs = Number.isInteger(options.timeoutMs) ? options.timeoutMs : 12000;

    let lastError = null;
    for (let attempt = 0; attempt <= retries; attempt += 1) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(url, {
                body: JSON.stringify(payload),
                headers: {
                    'Content-Type': 'application/json'
                },
                method: 'POST',
                signal: controller.signal,
            });
            clearTimeout(timer);
            return response;
        } catch (err) {
            clearTimeout(timer);
            lastError = err;
            if (attempt < retries) {
                await sleep(delayMs * (attempt + 1));
            }
        }
    }
    throw lastError || new Error('Unknown network error');
}

function negotiate() {
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // wait for ICE gathering to complete
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(async () => {
        var offer = pc.localDescription;
        const response = await postJsonWithRetry('/offer', {
            sdp: offer.sdp,
            type: offer.type,
        }, {
            retries: 2,
            delayMs: 800,
            timeoutMs: 15000,
        });
        if (!response.ok) {
            const bodyText = await response.text().catch(() => '');
            throw new Error('/offer failed: HTTP ' + response.status + ' ' + bodyText);
        }
        return response;
    }).then((response) => {
        return response.json();
    }).then((answer) => {
        document.getElementById('sessionid').value = answer.sessionid
        if (typeof window.onLiveTalkingSessionReady === 'function') {
            window.onLiveTalkingSessionReady(answer.sessionid);
        }
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        const message = (e && e.message) ? e.message : String(e);
        alert('Start session failed: ' + message);
    });
}

function start() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    pc = new RTCPeerConnection(config);

    // connect audio / video
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video') {
            document.getElementById('video').srcObject = evt.streams[0];
        } else {
            document.getElementById('audio').srcObject = evt.streams[0];
        }
    });

    document.getElementById('start').style.display = 'none';
    document.getElementById('stop').style.display = 'inline-block';
    return negotiate();
}

function stop() {
    var sid = document.getElementById('sessionid').value || '';
    if (typeof window.onLiveTalkingSessionStopped === 'function' && sid) {
        window.onLiveTalkingSessionStopped(sid);
    }
    document.getElementById('stop').style.display = 'none';

    // close peer connection
    setTimeout(() => {
        pc.close();
    }, 500);
}

window.onunload = function(event) {
    var sid = document.getElementById('sessionid').value || '';
    if (typeof window.onLiveTalkingSessionStopped === 'function' && sid) {
        window.onLiveTalkingSessionStopped(sid);
    }
    setTimeout(() => {
        pc.close();
    }, 500);
};

window.onbeforeunload = function (e) {
    var sid = document.getElementById('sessionid').value || '';
    if (typeof window.onLiveTalkingSessionStopped === 'function' && sid) {
        window.onLiveTalkingSessionStopped(sid);
    }
    setTimeout(() => {
        pc.close();
    }, 500);
    e = e || window.event;
    if (e) {
        e.returnValue = 'Đóng trang';
    }
    return 'Đóng trang';
}
