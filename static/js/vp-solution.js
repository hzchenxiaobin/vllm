/**
 * vp-solution.js
 * VitePress 风格独立页面（题解/文章）的渲染与交互脚本，
 * 与 vp-solution.css 配套使用（由 build/common.py 的 solution_page_template 生成页面）。
 *
 * 页面需先加载：marked.min.js、markdown-math.js、KaTeX、Prism（各语言组件）。
 * 本文件提供 window.VPPage.render(markdown)，完成：
 *   1. Markdown 渲染（标题锚点 + 悬停 "#" 链接，CJK 友好 slug）
 *   2. 代码块增强：语言标签 / 复制按钮 / 行号列
 *   3. 右侧「本页目录」生成 + 滚动高亮（outline-marker 跟随）
 *   4. 图片点击放大、外链新标签打开、回到顶部、明暗切换
 */
window.VPPage = (function () {
    'use strict';

    var COPY_SVG =
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" ' +
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
        'stroke-linejoin="round" aria-hidden="true">' +
        '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
        '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
        '</svg>';
    var CHECK_SVG =
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" ' +
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
        'stroke-linejoin="round" aria-hidden="true">' +
        '<polyline points="20 6 9 17 4 12"></polyline>' +
        '</svg>';

    function escapeHtml(text) {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    /* ------------------------------------------------------------------
     * Slug：CJK 友好（保留中日韩文字与字母数字），重复时追加 -1/-2
     * ------------------------------------------------------------------ */
    function makeSlugger() {
        var used = Object.create(null);
        return function (raw) {
            /* "Day N（...）：title" 概览标题映射为 day-N 锚点（跨页面 #day-N 链接） */
            var dayMatch = String(raw).match(/^Day\s+(\d+)/);
            if (dayMatch) return 'day-' + dayMatch[1];
            var slug = String(raw)
                .toLowerCase()
                .replace(/<[^>]+>/g, '')
                .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
                .replace(/[^\p{L}\p{N}\s_-]/gu, '')
                .trim()
                .replace(/\s+/g, '-')
                .replace(/-+/g, '-')
                .replace(/^-|-$/g, '');
            if (!slug) return '';
            if (/^[0-9]/.test(slug)) slug = '_' + slug;
            if (used[slug] != null) {
                slug = slug + '-' + (++used[slug]);
            } else {
                used[slug] = 0;
            }
            return slug;
        };
    }

    /* ------------------------------------------------------------------
     * Markdown 渲染
     * ------------------------------------------------------------------ */
    function renderMarkdown(markdownText) {
        var renderer = new marked.Renderer();
        var slugify = makeSlugger();

        renderer.heading = function (text, level, raw) {
            var anchor = slugify(raw);
            if (!anchor) {
                return '<h' + level + '>' + text + '</h' + level + '>';
            }
            return (
                '<h' + level + ' id="' + anchor + '">' + text +
                '<a class="header-anchor" href="#' + anchor + '" ' +
                'aria-label="永久链接到本标题"></a></h' + level + '>'
            );
        };

        marked.setOptions({
            renderer: renderer,
            headerIds: false,
            gfm: true,
            breaks: false,
            sanitize: false
        });

        document.getElementById('doc-content').innerHTML = marked.parse(markdownText);
    }

    /* ------------------------------------------------------------------
     * 代码块增强：包裹容器 + 语言标签 + 复制按钮 + 行号列
     * 须在 Prism.highlightAll() 之前调用（此时 code.textContent 即源码）。
     * ------------------------------------------------------------------ */
    function enhanceCodeBlocks() {
        document.querySelectorAll('.vp-doc pre > code').forEach(function (code) {
            var pre = code.parentNode;
            var match = (code.className || '').match(/language-([\w-]+)/);
            var lang = match ? match[1] : '';

            var container = document.createElement('div');
            container.className =
                'language-' + (lang || 'text') + ' line-numbers-mode';
            pre.parentNode.insertBefore(container, pre);
            container.appendChild(pre);

            var label = document.createElement('span');
            label.className = 'lang';
            label.textContent = lang || 'text';
            container.appendChild(label);

            var button = document.createElement('button');
            button.className = 'copy';
            button.type = 'button';
            button.title = 'Copy Code';
            button.innerHTML = COPY_SVG;
            button.addEventListener('click', function () {
                copyText(code.textContent, function () {
                    button.innerHTML = CHECK_SVG;
                    button.classList.add('copied');
                    setTimeout(function () {
                        button.innerHTML = COPY_SVG;
                        button.classList.remove('copied');
                    }, 2000);
                });
            });
            container.appendChild(button);

            var lineCount = code.textContent.replace(/\n+$/, '').split('\n').length;
            var numbers = [];
            for (var i = 1; i <= lineCount; i++) {
                numbers.push('<span class="line-number">' + i + '</span>');
            }
            var wrapper = document.createElement('div');
            wrapper.className = 'line-numbers-wrapper';
            wrapper.setAttribute('aria-hidden', 'true');
            wrapper.innerHTML = numbers.join('<br>');
            container.appendChild(wrapper);
        });
    }

    function copyText(text, onDone) {
        function fallback() {
            var ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            try {
                document.execCommand('copy');
                onDone();
            } catch (e) { /* ignore */ }
            document.body.removeChild(ta);
        }

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(onDone, fallback);
        } else {
            fallback();
        }
    }

    /* ------------------------------------------------------------------
     * 右侧「本页目录」+ 滚动高亮
     * ------------------------------------------------------------------ */
    function initOutline() {
        var doc = document.querySelector('.vp-doc');
        var outlineNav = document.querySelector('.vp-outline');
        if (!doc || !outlineNav) return;

        var headings = Array.prototype.slice.call(doc.querySelectorAll('h2, h3'));
        if (!headings.length || !headings.some(function (h) { return h.id; })) {
            outlineNav.style.display = 'none';
            return;
        }

        var root = document.getElementById('outline-root');
        var html = '';
        var seenH2 = false;
        headings.forEach(function (h) {
            if (!h.id) return;
            var nested = h.tagName === 'H3' && seenH2 ? ' nested' : '';
            if (h.tagName === 'H2') seenH2 = true;
            html +=
                '<li><a class="outline-link' + nested + '" href="#' + h.id + '">' +
                escapeHtml(h.textContent) + '</a></li>';
        });
        root.innerHTML = html;

        var links = Array.prototype.slice.call(root.querySelectorAll('.outline-link'));
        var marker = outlineNav.querySelector('.outline-marker');

        function atBottom() {
            return (
                window.innerHeight + window.scrollY >=
                document.documentElement.scrollHeight - 4
            );
        }

        function update() {
            var active = -1;
            for (var i = 0; i < headings.length; i++) {
                if (headings[i].getBoundingClientRect().top <= 96) active = i;
            }
            if (active === -1 && atBottom() && headings.length) {
                active = headings.length - 1;
            }
            links.forEach(function (link, i) {
                link.classList.toggle('active', i === active);
            });
            if (active >= 0 && marker) {
                marker.style.top = links[active].offsetTop + 7 + 'px';
                marker.style.opacity = 1;
            } else if (marker) {
                marker.style.opacity = 0;
            }
        }

        var ticking = false;
        window.addEventListener(
            'scroll',
            function () {
                if (ticking) return;
                ticking = true;
                requestAnimationFrame(function () {
                    update();
                    ticking = false;
                });
            },
            { passive: true }
        );
        window.addEventListener('resize', update);
        update();
    }

    /* ------------------------------------------------------------------
     * 明暗切换（与页首内联脚本共用 localStorage key：vp-theme）
     * ------------------------------------------------------------------ */
    function initThemeToggle() {
        var button = document.querySelector('.vp-appearance .vp-switch');
        if (!button) return;
        button.setAttribute('role', 'switch');

        function sync() {
            button.setAttribute(
                'aria-checked',
                document.documentElement.classList.contains('dark') ? 'true' : 'false'
            );
        }
        sync();

        button.addEventListener('click', function () {
            var dark = document.documentElement.classList.toggle('dark');
            try {
                localStorage.setItem('vp-theme', dark ? 'dark' : 'light');
            } catch (e) { /* ignore */ }
            sync();
        });
    }

    /* ------------------------------------------------------------------
     * 图片预览：点击放大，支持滚轮/按钮缩放、拖拽平移、双击切换
     * ------------------------------------------------------------------ */
    function initImageZoom() {
        var images = document.querySelectorAll('.vp-doc img');
        if (!images.length) return;

        var MIN_SCALE = 0.2, MAX_SCALE = 8;

        var lightbox = document.createElement('div');
        lightbox.className = 'vp-lightbox';
        lightbox.setAttribute('role', 'dialog');
        lightbox.setAttribute('aria-modal', 'true');
        lightbox.setAttribute('aria-label', '图片预览');
        lightbox.innerHTML =
            '<div class="lightbox-toolbar">' +
            '<button class="lightbox-btn" data-act="out" aria-label="缩小">&minus;</button>' +
            '<span class="lightbox-zoom">100%</span>' +
            '<button class="lightbox-btn" data-act="in" aria-label="放大">+</button>' +
            '<button class="lightbox-btn" data-act="reset" aria-label="适应窗口">&#x2922;</button>' +
            '<button class="lightbox-btn" data-act="close" aria-label="关闭预览">&times;</button>' +
            '</div>' +
            '<img src="" alt="">';
        document.body.appendChild(lightbox);

        var img = lightbox.querySelector('img');
        var zoomLabel = lightbox.querySelector('.lightbox-zoom');
        var scale = 1, tx = 0, ty = 0, dragging = false, startX = 0, startY = 0;

        function apply() {
            img.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
            zoomLabel.textContent = scale === 1 ? '适应' : Math.round(scale * 100) + '%';
        }

        function reset() {
            scale = 1;
            tx = 0;
            ty = 0;
            apply();
        }

        function zoomBy(factor) {
            scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale * factor));
            if (scale === 1) {
                tx = 0;
                ty = 0;
            }
            apply();
        }

        function close() {
            lightbox.classList.remove('active');
            document.body.style.overflow = '';
            reset();
        }

        images.forEach(function (image) {
            image.addEventListener('click', function (e) {
                e.preventDefault();
                img.src = image.src;
                img.alt = image.alt || '';
                reset();
                lightbox.classList.add('active');
                document.body.style.overflow = 'hidden';
            });
        });

        lightbox.querySelector('.lightbox-toolbar').addEventListener('click', function (e) {
            var act = e.target.getAttribute('data-act');
            if (act === 'in') zoomBy(1.25);
            else if (act === 'out') zoomBy(1 / 1.25);
            else if (act === 'reset') reset();
            else if (act === 'close') close();
        });

        /* 滚轮缩放 */
        lightbox.addEventListener('wheel', function (e) {
            e.preventDefault();
            zoomBy(e.deltaY < 0 ? 1.1 : 1 / 1.1);
        }, { passive: false });

        /* 拖拽平移 */
        img.addEventListener('pointerdown', function (e) {
            dragging = true;
            startX = e.clientX - tx;
            startY = e.clientY - ty;
            img.classList.add('dragging');
            img.setPointerCapture(e.pointerId);
        });
        img.addEventListener('pointermove', function (e) {
            if (!dragging) return;
            tx = e.clientX - startX;
            ty = e.clientY - startY;
            apply();
        });
        function endDrag() {
            dragging = false;
            img.classList.remove('dragging');
        }
        img.addEventListener('pointerup', endDrag);
        img.addEventListener('pointercancel', endDrag);

        /* 双击：适应窗口 ↔ 200% */
        img.addEventListener('dblclick', function (e) {
            e.preventDefault();
            if (scale === 1) {
                scale = 2;
                apply();
            } else {
                reset();
            }
        });

        /* 点击空白处关闭；点在图片上不关闭（避免与拖拽冲突） */
        lightbox.addEventListener('click', function (e) {
            if (e.target === lightbox) close();
        });
        document.addEventListener('keydown', function (e) {
            if (!lightbox.classList.contains('active')) return;
            if (e.key === 'Escape') close();
            else if (e.key === '+' || e.key === '=') zoomBy(1.25);
            else if (e.key === '-') zoomBy(1 / 1.25);
            else if (e.key === '0') reset();
        });
    }

    /* ------------------------------------------------------------------
     * 外链新标签打开（仅正文内的 http(s) 链接）
     * ------------------------------------------------------------------ */
    function initExternalLinks() {
        document.querySelectorAll('.vp-doc a[href^="http"]').forEach(function (a) {
            a.setAttribute('target', '_blank');
            a.setAttribute('rel', 'noopener noreferrer');
        });
    }

    /* ------------------------------------------------------------------
     * 回到顶部
     * ------------------------------------------------------------------ */
    function initBackToTop() {
        var button = document.querySelector('.vp-back-to-top');
        if (!button) return;
        window.addEventListener('scroll', function () {
            button.classList.toggle('visible', window.scrollY > 300);
        }, { passive: true });
        button.addEventListener('click', function () {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    /* ------------------------------------------------------------------
     * 胶囊导航：当前项滚动到可视区（移动端/长条）
     * ------------------------------------------------------------------ */
    function initPillBar() {
        var active = document.querySelector('.vp-nav-pills .vp-pill.active');
        if (active && typeof active.scrollIntoView === 'function') {
            active.scrollIntoView({ inline: 'center', block: 'nearest' });
        }
    }

    /* ------------------------------------------------------------------
     * 入口
     * ------------------------------------------------------------------ */
    function render(markdownText) {
        renderMarkdown(markdownText);
        /* 各增强步骤相互隔离：单个组件异常不影响页面其余部分 */
        [
            enhanceCodeBlocks,
            function () {
                if (window.Prism) Prism.highlightAll();
            },
            initOutline,
            initThemeToggle,
            initImageZoom,
            initExternalLinks,
            initBackToTop,
            initPillBar,
        ].forEach(function (step) {
            try {
                step();
            } catch (e) {
                console.error('VPPage step failed:', e);
            }
        });
    }

    return { render: render };
})();
