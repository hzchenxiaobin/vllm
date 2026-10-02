/**
 * kimi-chat.js
 * 全站每页的「咨询本篇内容」悬浮窗口：把当前页面 Markdown 全文（window.pageMarkdown）
 * 作为上下文，调用 Moonshot（Kimi）OpenAI 兼容接口流式回答用户关于本篇内容的问题。
 *
 * 依赖（可选，缺失自动降级）：marked（渲染回复 Markdown）、KaTeX（渲染公式）、
 * Prism（代码高亮）。API Key / 模型 / 接口地址在窗口设置中配置，保存在 localStorage。
 */
(function () {
    'use strict';

    if (window.__KIMI_CHAT_LOADED__) return;
    window.__KIMI_CHAT_LOADED__ = true;

    var LS_KEY = 'kc.settings.v1';
    var DEFAULTS = {
        apiKey: '',
        baseURL: 'https://api.moonshot.ai/v1',
        model: 'kimi-k3',
        maxContextChars: 100000
    };
    var TEMPERATURE = 0.6;
    var MAX_TOKENS = 4096;
    var HISTORY_LIMIT = 12;
    var QUICK_QUESTIONS = [
        '总结本篇的核心要点',
        '解释本篇最重要的公式与推导',
        '本篇内容面试会怎么考？',
        '给我出 3 道自测题（附答案）'
    ];

    var els = {};
    var settings = loadSettings();
    var messages = [];
    var streaming = false;
    var abortCtrl = null;
    var welcomeHTML = '';
    var errorTimer = null;

    function $(id) {
        return document.getElementById(id);
    }

    /* ------------------------------------------------------------------
     * 设置读写
     * ------------------------------------------------------------------ */

    function loadSettings() {
        var saved = {};
        try {
            saved = JSON.parse(localStorage.getItem(LS_KEY) || '{}') || {};
        } catch (e) {
            saved = {};
        }
        var out = {};
        for (var key in DEFAULTS) {
            if (!Object.prototype.hasOwnProperty.call(DEFAULTS, key)) continue;
            var val = saved[key];
            if (key === 'apiKey') {
                out[key] = typeof val === 'string' ? val : '';
            } else if (key === 'maxContextChars') {
                var num = parseInt(val, 10);
                out[key] = isFinite(num) && num >= 2000 ? num : DEFAULTS[key];
            } else {
                out[key] = typeof val === 'string' && val.trim() ? val.trim() : DEFAULTS[key];
            }
        }
        return out;
    }

    function persistSettings() {
        try {
            localStorage.setItem(LS_KEY, JSON.stringify(settings));
        } catch (e) {
            console.warn('[kimi-chat] 无法保存设置到 localStorage：', e);
        }
    }

    /* ------------------------------------------------------------------
     * 页面上下文
     * ------------------------------------------------------------------ */

    function getArticleMarkdown() {
        var md = window.pageMarkdown;
        if (typeof md === 'string' && md.trim()) return md;
        var node = document.querySelector('.vp-doc') || document.querySelector('main') ||
            document.querySelector('article');
        if (node) return node.innerText || '';
        return '';
    }

    function buildSystemPrompt() {
        var title = (document.title || '').replace(/\s+/g, ' ').trim();
        var md = getArticleMarkdown();
        var truncated = '';
        if (md.length > settings.maxContextChars) {
            md = md.slice(0, settings.maxContextChars);
            truncated = '\n（注：正文过长，以上内容已截断。）';
        }
        return [
            '你是学习网站「vLLM Notes」的页面助教，由 Kimi 大模型驱动，负责解答用户关于当前页面文章的问题。',
            '当前页面标题：' + (title || '（未知）'),
            '当前页面的 Markdown 正文如下（其中的图片 / SVG 图无法查看）：',
            '<<<PAGE_CONTENT',
            md,
            'PAGE_CONTENT>>>' + truncated,
            '',
            '回答要求：',
            '1. 使用简体中文，语气专业、友好；',
            '2. 优先依据上文页面内容回答；页面未涉及的内容可结合你自己的知识补充，但要注明「页面未提及，以下为补充」；',
            '3. 回答简明、结构化，合理使用 Markdown 列表 / 表格 / 代码块；',
            '4. 涉及公式时使用 $...$ 或 $$...$$ 语法。'
        ].join('\n');
    }

    function buildRequestMessages() {
        var history = messages.filter(function (m) {
            return m.content;
        }).slice(-HISTORY_LIMIT);
        return [{ role: 'system', content: buildSystemPrompt() }].concat(history);
    }

    /* ------------------------------------------------------------------
     * 渲染
     * ------------------------------------------------------------------ */

    function escapeHtml(text) {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function renderMarkdown(text) {
        if (typeof marked !== 'undefined' && marked.parse) {
            try {
                return marked.parse(text || '');
            } catch (e) {
                console.warn('[kimi-chat] marked 解析失败：', e);
            }
        }
        return '<p>' + escapeHtml(text || '').replace(/\n/g, '<br>') + '</p>';
    }

    function renderMathIn(el) {
        if (typeof katex === 'undefined') return;
        el.querySelectorAll('.math-inline').forEach(function (node) {
            try {
                katex.render(node.dataset.latex || '', node, {
                    throwOnError: false,
                    displayMode: false
                });
            } catch (e) { /* 单个公式失败不影响整体 */ }
        });
        el.querySelectorAll('.math-block').forEach(function (node) {
            try {
                katex.render(node.dataset.latex || '', node, {
                    throwOnError: false,
                    displayMode: true
                });
            } catch (e) { /* 同上 */ }
        });
    }

    function highlightCodeIn(el) {
        if (typeof Prism === 'undefined' || !Prism.highlightElement) return;
        el.querySelectorAll('pre code[class*="language-"]').forEach(function (code) {
            try {
                Prism.highlightElement(code);
            } catch (e) { /* 高亮失败按纯文本显示 */ }
        });
    }

    function renderBubble(el, text, finalize) {
        el.innerHTML = renderMarkdown(text);
        renderMathIn(el);
        if (finalize) highlightCodeIn(el);
    }

    function scheduleRender(bubble, getText) {
        if (bubble.__raf) return;
        bubble.__raf = requestAnimationFrame(function () {
            bubble.__raf = null;
            renderBubble(bubble, getText(), false);
            scrollBody();
        });
    }

    function appendBubble(role, text) {
        var msg = document.createElement('div');
        msg.className = 'kc-msg ' + role;

        var avatar = document.createElement('span');
        avatar.className = 'kc-avatar';
        avatar.textContent = role === 'user' ? '🧑‍💻' : '🤖';

        var bubble = document.createElement('div');
        bubble.className = 'kc-bubble';
        bubble.__msgNode = msg;
        if (role === 'user') {
            bubble.textContent = text;
        } else {
            bubble.innerHTML = '<span class="kc-typing"><i></i><i></i><i></i></span>';
        }

        msg.appendChild(avatar);
        msg.appendChild(bubble);
        els.body.appendChild(msg);
        scrollBody();
        return bubble;
    }

    function scrollBody() {
        els.body.scrollTop = els.body.scrollHeight;
    }

    function removeWelcome() {
        var welcome = els.body.querySelector('.kc-welcome');
        if (welcome) welcome.remove();
    }

    function buildChips() {
        var chips = els.body.querySelector('#kc-chips');
        if (!chips) return;
        chips.innerHTML = '';
        QUICK_QUESTIONS.forEach(function (q) {
            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'kc-chip';
            btn.textContent = q;
            btn.addEventListener('click', function () {
                sendMessage(q);
            });
            chips.appendChild(btn);
        });
    }

    /* ------------------------------------------------------------------
     * 提示条
     * ------------------------------------------------------------------ */

    function showError(msg) {
        els.error.textContent = msg;
        els.error.classList.remove('info');
        els.error.hidden = false;
        if (errorTimer) clearTimeout(errorTimer);
        errorTimer = setTimeout(function () {
            els.error.hidden = true;
        }, 10000);
    }

    function showNotice(msg) {
        els.error.textContent = msg;
        els.error.classList.add('info');
        els.error.hidden = false;
        if (errorTimer) clearTimeout(errorTimer);
        errorTimer = setTimeout(function () {
            els.error.hidden = true;
        }, 4000);
    }

    function hideError() {
        els.error.hidden = true;
        if (errorTimer) {
            clearTimeout(errorTimer);
            errorTimer = null;
        }
    }

    function setStatus(visible, text) {
        els.status.hidden = !visible;
        if (visible && text) els.status.textContent = text;
    }

    function setStreaming(flag) {
        streaming = flag;
        els.sendBtn.hidden = flag;
        els.stopBtn.hidden = !flag;
    }

    /* ------------------------------------------------------------------
     * 请求
     * ------------------------------------------------------------------ */

    function joinURL(base, path) {
        return String(base || '').replace(/\/+$/, '') + path;
    }

    function readStream(body, onDelta) {
        return new Promise(function (resolve, reject) {
            var reader = body.getReader();
            var decoder = new TextDecoder('utf-8');
            var buf = '';

            function handleLine(line) {
                line = line.replace(/\r$/, '').trim();
                if (!line || line.indexOf('data:') !== 0) return;
                var payload = line.slice(5).trim();
                if (payload === '[DONE]') return;
                try {
                    var json = JSON.parse(payload);
                    var delta = json.choices && json.choices[0] && json.choices[0].delta;
                    if (!delta) return;
                    if (delta.reasoning_content) onDelta(null, delta.reasoning_content);
                    if (delta.content) onDelta(delta.content, null);
                } catch (e) { /* 忽略心跳等非 JSON 行 */ }
            }

            function pump() {
                reader.read().then(function (chunk) {
                    if (chunk.done) {
                        resolve();
                        return;
                    }
                    buf += decoder.decode(chunk.value, { stream: true });
                    var idx;
                    while ((idx = buf.indexOf('\n')) !== -1) {
                        handleLine(buf.slice(0, idx));
                        buf = buf.slice(idx + 1);
                    }
                    pump();
                }).catch(reject);
            }

            pump();
        });
    }

    function httpError(res, bodyText) {
        var msg = '请求失败（HTTP ' + res.status + '）';
        if (bodyText) {
            try {
                var data = JSON.parse(bodyText);
                if (data && data.error && data.error.message) {
                    msg = data.error.message;
                }
            } catch (e) {
                msg += '：' + bodyText.slice(0, 200);
            }
        }
        if (res.status === 401) msg = 'API Key 无效或未填写。' + msg;
        if (res.status === 404) msg += '（请检查 API 地址与模型名称是否正确）';
        return new Error(msg);
    }

    function finish(reply, bubble, full, err) {
        setStreaming(false);
        setStatus(false);
        reply.content = full || '';

        if (!reply.content) {
            messages.pop();
            if (bubble.__msgNode) bubble.__msgNode.remove();
            if (err && err.name !== 'AbortError') {
                showError(err.message || '请求失败，请稍后重试。');
            } else if (!err) {
                showError('模型未返回内容，请重试或在设置中更换模型。');
            }
            return;
        }

        renderBubble(bubble, reply.content, true);
        if (bubble.__raf) {
            cancelAnimationFrame(bubble.__raf);
            bubble.__raf = null;
        }
        if (err && err.name === 'AbortError') {
            var note = document.createElement('div');
            note.className = 'kc-stopped-note';
            note.textContent = '（已停止生成）';
            bubble.appendChild(note);
        }
        scrollBody();
    }

    function sendMessage(text) {
        text = String(text || '').trim();
        if (!text || streaming) return;

        if (!settings.apiKey) {
            showError('请先点击右上角 ⚙️ 填写 Moonshot API Key（只需一次，保存在本浏览器中）。');
            openSettings();
            return;
        }

        hideError();
        removeWelcome();
        messages.push({ role: 'user', content: text });
        appendBubble('user', text);

        var reply = { role: 'assistant', content: '' };
        messages.push(reply);
        var bubble = appendBubble('assistant', '');
        setStatus(true, '正在思考');
        setStreaming(true);

        var full = '';
        abortCtrl = new AbortController();

        fetch(joinURL(settings.baseURL, '/chat/completions'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + settings.apiKey
            },
            body: JSON.stringify({
                model: settings.model,
                messages: buildRequestMessages(),
                stream: true,
                temperature: TEMPERATURE,
                max_tokens: MAX_TOKENS
            }),
            signal: abortCtrl.signal
        }).then(function (res) {
            if (!res.ok) {
                return res.text().then(function (bodyText) {
                    throw httpError(res, bodyText);
                });
            }
            if (!res.body || typeof res.body.getReader !== 'function') {
                return res.json().then(function (data) {
                    var content = data && data.choices && data.choices[0] &&
                        data.choices[0].message && data.choices[0].message.content;
                    full = content || '';
                });
            }
            return readStream(res.body, function (delta, reasoning) {
                if (reasoning) {
                    setStatus(true, '正在深度思考');
                    return;
                }
                if (delta) {
                    full += delta;
                    setStatus(false);
                    scheduleRender(bubble, function () {
                        return full;
                    });
                }
            });
        }).then(function () {
            finish(reply, bubble, full, null);
        }).catch(function (err) {
            finish(reply, bubble, full, err);
        });
    }

    function stopStream() {
        if (abortCtrl) {
            try {
                abortCtrl.abort();
            } catch (e) { /* 已结束 */ }
        }
    }

    /* ------------------------------------------------------------------
     * 面板 / 设置交互
     * ------------------------------------------------------------------ */

    function openPanel() {
        els.panel.classList.add('open');
        els.panel.setAttribute('aria-hidden', 'false');
        setTimeout(function () {
            els.input.focus();
        }, 60);
    }

    function closePanel() {
        els.panel.classList.remove('open');
        els.panel.setAttribute('aria-hidden', 'true');
    }

    function togglePanel() {
        if (els.panel.classList.contains('open')) {
            closePanel();
        } else {
            openPanel();
        }
    }

    function openSettings() {
        els.setKey.value = settings.apiKey;
        els.setModel.value = settings.model;
        els.setBase.value = settings.baseURL;
        els.setMaxCtx.value = String(settings.maxContextChars);
        els.settings.hidden = false;
        els.setKey.focus();
    }

    function closeSettings() {
        els.settings.hidden = true;
    }

    function saveSettings() {
        settings.apiKey = els.setKey.value.trim();
        settings.model = els.setModel.value.trim() || DEFAULTS.model;
        settings.baseURL = els.setBase.value.trim() || DEFAULTS.baseURL;
        var cap = parseInt(els.setMaxCtx.value, 10);
        settings.maxContextChars = isFinite(cap) && cap >= 2000 ? cap : DEFAULTS.maxContextChars;
        persistSettings();
        updateChrome();
        closeSettings();
        showNotice('设置已保存。');
    }

    function clearChat() {
        if (streaming) stopStream();
        messages = [];
        els.body.innerHTML = welcomeHTML;
        buildChips();
        hideError();
        setStatus(false);
    }

    function updateChrome() {
        els.modelBadge.textContent = settings.model;
        var title = (document.title || '').trim();
        els.pageInfo.textContent = '📖 ' + (title || '当前页面');
        els.pageInfo.title = title;
    }

    function autosize() {
        els.input.style.height = 'auto';
        els.input.style.height = Math.min(els.input.scrollHeight, 120) + 'px';
    }

    /* ------------------------------------------------------------------
     * 初始化
     * ------------------------------------------------------------------ */

    function bindEvents() {
        els.launch.addEventListener('click', togglePanel);
        els.closeBtn.addEventListener('click', closePanel);
        els.settingsBtn.addEventListener('click', openSettings);
        els.settingsClose.addEventListener('click', closeSettings);
        els.saveBtn.addEventListener('click', saveSettings);
        els.clearBtn.addEventListener('click', clearChat);
        els.sendBtn.addEventListener('click', function () {
            var text = els.input.value;
            els.input.value = '';
            autosize();
            sendMessage(text);
        });
        els.stopBtn.addEventListener('click', stopStream);

        els.input.addEventListener('input', autosize);
        els.input.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
                e.preventDefault();
                var text = els.input.value;
                els.input.value = '';
                autosize();
                sendMessage(text);
            }
        });

        document.addEventListener('keydown', function (e) {
            if (e.key !== 'Escape') return;
            if (!els.settings.hidden) {
                closeSettings();
            } else if (els.panel.classList.contains('open')) {
                closePanel();
            }
        });

        els.body.addEventListener('click', function (e) {
            var link = e.target.closest('a[href^="http"]');
            if (link) {
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
            }
        });
    }

    function init() {
        els.launch = $('kc-launch');
        els.panel = $('kc-panel');
        if (!els.launch || !els.panel) return;
        els.modelBadge = $('kc-model-badge');
        els.pageInfo = $('kc-page-info');
        els.body = $('kc-body');
        els.error = $('kc-error');
        els.status = $('kc-status');
        els.input = $('kc-input');
        els.sendBtn = $('kc-send-btn');
        els.stopBtn = $('kc-stop-btn');
        els.settingsBtn = $('kc-settings-btn');
        els.closeBtn = $('kc-close-btn');
        els.settings = $('kc-settings');
        els.settingsClose = $('kc-settings-close');
        els.setKey = $('kc-set-key');
        els.setModel = $('kc-set-model');
        els.setBase = $('kc-set-base');
        els.setMaxCtx = $('kc-set-maxctx');
        els.saveBtn = $('kc-save-btn');
        els.clearBtn = $('kc-clear-btn');
        if (!els.body || !els.input || !els.settings || !els.sendBtn) return;

        welcomeHTML = els.body.innerHTML;
        buildChips();
        updateChrome();
        bindEvents();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
