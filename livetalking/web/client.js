var pc = null;
var remoteStream = null;

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
    }).then(() => {
        var offer = pc.localDescription;
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        return response.json();
    }).then((answer) => {
        document.getElementById('sessionid').value = answer.sessionid
        if (typeof window.onWebRTCSessionCreated === 'function') {
            window.onWebRTCSessionCreated(answer.sessionid);
        }
        return pc.setRemoteDescription(answer);
    }).then(() => {
        if (typeof window.onWebRTCConnected === 'function') {
            window.onWebRTCConnected();
        }
    }).catch((e) => {
        alert(e);
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
    pc.addEventListener('connectionstatechange', () => {
        if (pc.connectionState === 'connected' && typeof window.onWebRTCConnected === 'function') {
            window.onWebRTCConnected();
        }
        if ((pc.connectionState === 'failed' || pc.connectionState === 'closed' || pc.connectionState === 'disconnected')
            && typeof window.onWebRTCDisconnected === 'function') {
            window.onWebRTCDisconnected();
        }
    });

    // connect audio / video
    remoteStream = new MediaStream();
    pc.addEventListener('track', (evt) => {
        const videoEl = document.getElementById('video');
        const audioEl = document.getElementById('audio');

        // Some browsers send tracks without evt.streams[0], so build a stream manually.
        remoteStream.addTrack(evt.track);

        if (videoEl) {
            videoEl.srcObject = remoteStream;
            videoEl.muted = false;
            videoEl.play().catch(() => {});
        }
        if (audioEl) {
            audioEl.srcObject = remoteStream;
            audioEl.muted = false;
            audioEl.play().catch(() => {});
        }
    });

    document.getElementById('start').style.display = 'none';
    negotiate();
    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    document.getElementById('stop').style.display = 'none';
    document.getElementById('sessionid').value = '0';

    // close peer connection
    setTimeout(() => {
        pc.close();
        if (typeof window.onWebRTCDisconnected === 'function') {
            window.onWebRTCDisconnected();
        }
    }, 500);
}

window.onunload = function(event) {
    // 在这里执行你想要的操作
    setTimeout(() => {
        pc.close();
    }, 500);
};

window.onbeforeunload = function (e) {
        setTimeout(() => {
                pc.close();
            }, 500);
        e = e || window.event
        // 兼容IE8和Firefox 4之前的版本
        if (e) {
          e.returnValue = '关闭提示'
        }
        // Chrome, Safari, Firefox 4+, Opera 12+ , IE 9+
        return '关闭提示'
      }
