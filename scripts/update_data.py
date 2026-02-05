#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Update data.json for GitHub Pages (GitHub Actions).

Goal
- Produce a daily web-ready list of AI hotspots (<=10 items) for the last 24–36 hours.
- Prefer *detailed content shown directly* in the card.

Strategy (B first, fallback to A)
- B: fetch original article page -> extract readable text -> build a longer summary/excerpt.
- A: if fetch/extract fails -> use RSS description/summary (when available).

Sources (no API keys)
- Hacker News front page (HTML): title + url + points (hotness signal)
- 机器之心 RSS (中文)
- The Verge RSS (Atom)
- MIT Technology Review RSS

Notes
- No LLM calls are used in this script (stable, free). English pieces are wrapped in Chinese
  and may include short English excerpts. Proper nouns remain.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Optional, List, Dict, Tuple

import urllib.request


AI_KEYWORDS = [
    "ai",
    "llm",
    "agent",
    "agents",
    "anthropic",
    "openai",
    "claude",
    "copilot",
    "mistral",
    "transcribe",
    "model",
    "inference",
    "alignment",
    "safety",
    "regulation",
    "copyright",
    "deepfake",
]


def fetch(url: str, timeout: int = 25) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; BigAIEventBot/1.0; +https://github.com/zhouhui88/BigAIEvent)",
            "Accept": "text/html,application/xml,application/rss+xml,application/atom+xml,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def now_shanghai() -> datetime:
    # Simple fixed offset +8 (good enough for daily page title)
    return datetime.utcnow() + timedelta(hours=8)


def within_window(dt: Optional[datetime], *, start: datetime, end: datetime) -> bool:
    if dt is None:
        return False
    return start <= dt <= end


def normalize_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def html_to_text(html: str, limit_chars: int = 4000) -> str:
    # unwrap CDATA
    html = re.sub(r"^\s*<!\[CDATA\[", "", html)
    html = re.sub(r"\]\]>\s*$", "", html)

    # remove scripts/styles
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)

    # drop common boilerplate blocks
    html = re.sub(r"<nav[\s\S]*?</nav>", " ", html, flags=re.I)
    html = re.sub(r"<header[\s\S]*?</header>", " ", html, flags=re.I)
    html = re.sub(r"<footer[\s\S]*?</footer>", " ", html, flags=re.I)

    # replace breaks
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    html = re.sub(r"</p>|</div>|</li>|</h\d>", "\n", html, flags=re.I)

    # First unescape so escaped HTML becomes real tags
    html = unescape(html)

    # strip tags (pass 1)
    text = re.sub(r"<[^>]+>", " ", html)
    # unescape again in case of nested escaping
    text = unescape(text)
    # strip tags (pass 2)
    text = re.sub(r"<[^>]+>", " ", text)

    # normalize whitespace
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if len(text) > limit_chars:
        text = text[:limit_chars]
    return text


def _extract_main_html(raw: str) -> str:
    # Prefer <article> or <main> blocks to reduce navigation noise
    m = re.search(r"<article[\s\S]*?</article>", raw, flags=re.I)
    if m:
        return m.group(0)
    m = re.search(r"<main[\s\S]*?</main>", raw, flags=re.I)
    if m:
        return m.group(0)
    return raw


def _cleanup_excerpt(text: str) -> str:
    # Remove common boilerplate phrases
    drop_phrases = [
        "skip to main content",
        "skip to content",
        "skip to footer",
        "menu and widgets",
        "subscribe",
        "privacy",
        "terms",
    ]
    tl = text
    for p in drop_phrases:
        tl = re.sub(re.escape(p), " ", tl, flags=re.I)

    # Remove repeated UI words
    tl = re.sub(r"\b(products?|solutions?|pricing|company|resources?)\b", " ", tl, flags=re.I)
    tl = normalize_ws(tl)

    # Keep a compact excerpt
    return tl[:600]


def extract_excerpt_from_url(url: str) -> Optional[str]:
    try:
        raw = fetch(url, timeout=20)
    except Exception:
        return None

    main_html = _extract_main_html(raw)
    text = html_to_text(main_html, limit_chars=8000)
    text = _cleanup_excerpt(text)

    if len(text) < 160:
        return None

    return text


@dataclass
class Item:
    id: str
    lang: str  # 'en'|'zh'
    type: str  # 'product'|'policy'
    hotness: int
    title: str
    summary: str
    sources: List[Dict[str, str]]
    published_at: Optional[datetime] = None


def classify_type(text: str) -> str:
    t = text.lower()
    policy_keys = ["policy", "regulation", "security", "safety", "govern", "copyright", "lawsuit", "espionage"]
    if any(k in t for k in policy_keys):
        return "policy"
    return "product"


def zh_template(title: str) -> str:
    tl = title.lower()
    if "voxtral" in tl and "transcribe" in tl:
        return "Mistral 更新语音转写（speech-to-text）相关能力。"
    if "space to think" in tl:
        return "Anthropic 发布文章解释 Claude 的产品定位（space to think）。"
    if "copilot" in tl and "problem" in tl:
        return "媒体报道 Microsoft Copilot 在产品落地/体验上遇到挑战。"
    if "guardrails" in tl and "governance" in tl and "agent" in tl:
        return "讨论如何用“边界治理”而非纯提示词来管控 agentic systems。"
    if "ai is killing" in tl and "saas" in tl:
        return "观点文章讨论 AI 对传统 B2B SaaS 的冲击。"
    if "connect to a local model" in tl and "claude code" in tl:
        return "经验帖：Claude Code 额度用尽时，如何切换连接本地开源模型继续工作。"
    if "claude code for infrastructure" in tl:
        return "工具介绍：用 Claude Code 辅助基础设施运维（先在沙箱/克隆环境验证，再生成 IaC）。"
    if "benchmark" in tl and "code review" in tl:
        return "基准/评测：提出面向 AI 代码审查（code review）的真实场景评测。"
    return ""


def build_detailed_summary(*, title: str, lang: str, original_url: Optional[str], rss_fallback: Optional[str]) -> str:
    """B first: fetch original and show a longer excerpt; fallback to A: RSS summary."""

    base = zh_template(title)

    excerpt = None
    if original_url:
        excerpt = extract_excerpt_from_url(original_url)

    if excerpt:
        # Prefer Chinese framing + extract highlights.
        head = base if base else ("" if lang == "zh" else f"英文资讯：{title}。")

        highlights = []
        exl = excerpt.lower()
        if "open-weights" in exl or "open weights" in exl:
            highlights.append("开源权重（open-weights）")
        if "apache" in exl:
            highlights.append("Apache 2.0 许可")
        if "real-time" in exl or "realtime" in exl:
            highlights.append("支持实时（real-time）")
        if "diarization" in exl:
            highlights.append("支持说话人分离（diarization）")
        if "low latency" in exl:
            highlights.append("低延迟（low latency）")
        if "ad-free" in exl or "ad free" in exl:
            highlights.append("明确不引入广告（ad-free）")
        if "infrastructure" in exl or "terraform" in exl or "ansible" in exl:
            highlights.append("面向基础设施/IaC（Terraform/Ansible）")
        if "local model" in exl or "open source model" in exl:
            highlights.append("可切换本地/开源模型")

        hl_line = ("亮点：" + "、".join(highlights)) if highlights else ""

        # For EN: do NOT dump raw English excerpt into the card (too noisy). Use highlights only.
        if lang == "en":
            parts = [p for p in [head, hl_line] if p]
            return normalize_ws("\n".join(parts))

        # For ZH: show a compact excerpt (already Chinese)
        parts = [p for p in [head, hl_line] if p]
        return normalize_ws("\n".join(parts) + ("\n\n要点摘录：\n" + excerpt if excerpt else ""))

    if rss_fallback:
        fb = normalize_ws(html_to_text(rss_fallback, limit_chars=800))
        if fb:
            head = base if base else ("" if lang == "zh" else f"英文资讯：{title}。")
            return normalize_ws(head + ("\n\n" if head else "") + fb)

    # last resort (Chinese)
    if base:
        return base + "（建议点来源查看原文）"
    if lang == "en":
        return f"英文报道：{title}（建议点来源查看原文）"
    return "详见来源。"


def parse_hn() -> List[Item]:
    html = fetch("https://news.ycombinator.com/")

    titles: List[Tuple[str, str]] = []
    for m in re.finditer(r'<span class="titleline">\s*<a href="(.*?)"[^>]*>(.*?)</a>', html, re.S):
        url = unescape(m.group(1)).strip()
        title = unescape(re.sub(r"<.*?>", "", m.group(2))).strip()
        titles.append((title, url))

    scores: List[int] = []
    for m in re.finditer(r'<span class="score"[^>]*>(\d+) points</span>', html):
        scores.append(int(m.group(1)))

    items: List[Item] = []
    for idx, (title, url) in enumerate(titles[:30]):
        pts = scores[idx] if idx < len(scores) else 0
        tl = title.lower()
        if not any(k in tl for k in AI_KEYWORDS):
            continue

        item_id = f"hn-{idx}-{re.sub(r'[^a-z0-9]+','-',tl)[:40]}".strip("-")
        itype = classify_type(title)

        summary = build_detailed_summary(title=title, lang="en", original_url=url, rss_fallback=None)

        items.append(
            Item(
                id=item_id,
                lang="en",
                type=itype,
                hotness=min(100, 40 + pts // 10),
                title=title,
                summary=summary,
                sources=[
                    {"name": "Hacker News", "url": "https://news.ycombinator.com/"},
                    {"name": "原文", "url": url},
                ],
                published_at=None,
            )
        )

    return items


def parse_rss_items_with_desc(xml: str) -> List[Tuple[str, str, Optional[datetime], str]]:
    out = []
    for m in re.finditer(
        r"<item>.*?<title>(.*?)</title>.*?<link>(.*?)</link>.*?<pubDate>(.*?)</pubDate>(.*?)</item>",
        xml,
        re.S,
    ):
        title = unescape(re.sub(r"<.*?>", "", m.group(1))).strip()
        link = unescape(m.group(2)).strip()
        pub_raw = m.group(3).strip()
        tail = m.group(4)

        desc = ""
        m_desc = re.search(r"<description>(.*?)</description>", tail, re.S)
        if m_desc:
            desc = m_desc.group(1)
        m_content = re.search(r"<content:encoded>(.*?)</content:encoded>", tail, re.S)
        if m_content and len(m_content.group(1)) > len(desc):
            desc = m_content.group(1)

        dt = None
        try:
            dt = parsedate_to_datetime(pub_raw)
            dt = dt.astimezone(tz=None).replace(tzinfo=None)
        except Exception:
            dt = None

        out.append((title, link, dt, desc))

    return out


def parse_atom_entries(xml: str) -> List[Tuple[str, str, Optional[datetime], List[str], str]]:
    out = []
    for m in re.finditer(
        r"<entry>.*?<title[^>]*><!\[CDATA\[(.*?)\]\]></title>.*?<link rel=\"alternate\"[^>]*href=\"(.*?)\"[^>]*/>.*?<published>(.*?)</published>(.*?)</entry>",
        xml,
        re.S,
    ):
        title = unescape(m.group(1)).strip()
        link = unescape(m.group(2)).strip()
        published_raw = m.group(3).strip()
        tail = m.group(4)
        cats = re.findall(r"<category[^>]*term=\"(.*?)\"", tail)
        summary = ""
        m_sum = re.search(r"<summary[^>]*><!\[CDATA\[(.*?)\]\]></summary>", tail, re.S)
        if m_sum:
            summary = m_sum.group(1)

        dt = None
        try:
            dt = datetime.fromisoformat(published_raw.replace("Z", "+00:00")).astimezone(tz=None).replace(tzinfo=None)
        except Exception:
            dt = None

        out.append((title, link, dt, cats, summary))

    return out


def parse_jiqizhixin(start: datetime, end: datetime) -> List[Item]:
    xml = fetch("https://www.jiqizhixin.com/rss")
    rows = parse_rss_items_with_desc(xml)

    items: List[Item] = []
    for i, (title, link, dt_local, desc) in enumerate(rows):
        if dt_local is not None and not within_window(dt_local, start=start, end=end):
            continue

        detailed = build_detailed_summary(title=title, lang="zh", original_url=link, rss_fallback=desc)
        if not detailed or detailed == "详见来源。":
            detailed = "（中文媒体）详见来源。"

        items.append(
            Item(
                id=f"jiqizhixin-{i}",
                lang="zh",
                type=classify_type(title),
                hotness=60,
                title=title,
                summary=detailed,
                sources=[{"name": "机器之心", "url": link}],
                published_at=dt_local,
            )
        )

    return items


def parse_verge(start: datetime, end: datetime) -> List[Item]:
    xml = fetch("https://www.theverge.com/rss/index.xml")
    entries = parse_atom_entries(xml)

    items: List[Item] = []
    for i, (title, link, dt_utc_naive, cats, sum_html) in enumerate(entries):
        if "AI" not in cats and not any(k in title.lower() for k in AI_KEYWORDS):
            continue

        # Approximate to Asia/Shanghai
        dt_sh = None
        if dt_utc_naive is not None:
            dt_sh = dt_utc_naive + timedelta(hours=8)

        if dt_sh is not None and not within_window(dt_sh, start=start, end=end):
            continue

        detailed = build_detailed_summary(title=title, lang="en", original_url=link, rss_fallback=sum_html)

        items.append(
            Item(
                id=f"verge-{i}",
                lang="en",
                type=classify_type(title),
                hotness=55,
                title=title,
                summary=detailed,
                sources=[{"name": "The Verge", "url": link}],
                published_at=dt_sh,
            )
        )

    return items


def parse_mit_tr(start: datetime, end: datetime) -> List[Item]:
    xml = fetch("https://www.technologyreview.com/feed/")
    rows = parse_rss_items_with_desc(xml)

    items: List[Item] = []
    for i, (title, link, dt_utc_naive, desc) in enumerate(rows):
        if not any(k in title.lower() for k in AI_KEYWORDS + ["agentic"]):
            continue

        dt_sh = None
        if dt_utc_naive is not None:
            dt_sh = dt_utc_naive + timedelta(hours=8)

        if dt_sh is not None and not within_window(dt_sh, start=start, end=end):
            continue

        detailed = build_detailed_summary(title=title, lang="en", original_url=link, rss_fallback=desc)

        items.append(
            Item(
                id=f"mittr-{i}",
                lang="en",
                type="policy",
                hotness=50,
                title=title,
                summary=detailed,
                sources=[{"name": "MIT Technology Review", "url": link}],
                published_at=dt_sh,
            )
        )

    return items


def dedupe(items: List[Item]) -> List[Item]:
    seen = set()
    out: List[Item] = []
    for it in items:
        key = re.sub(r"\W+", " ", it.title.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def main() -> int:
    now = now_shanghai()
    end = now
    start = now - timedelta(hours=36)
    min_age = now - timedelta(hours=24)

    collected: List[Item] = []

    # Sources may fail; keep going.
    for fn in (parse_hn,):
        try:
            collected.extend(fn())
        except Exception as e:
            print(f"WARN: {fn.__name__} failed: {e}", file=sys.stderr)

    for fn in (parse_jiqizhixin, parse_verge, parse_mit_tr):
        try:
            collected.extend(fn(start, end))
        except Exception as e:
            print(f"WARN: {fn.__name__} failed: {e}", file=sys.stderr)

    collected = dedupe(collected)

    # Filter by min_age when we have timestamps; otherwise keep but down-rank.
    for it in collected:
        if it.published_at is not None and it.published_at > min_age:
            it.hotness = max(1, it.hotness - 15)

    collected.sort(key=lambda x: (x.hotness or 0), reverse=True)
    top = collected[:10]

    display_title = f"{now.year}年{now.month}月{now.day}日 AI大事件"

    data = {
        "displayTitle": display_title,
        "title": "AI大事件",
        "generatedAt": now.strftime("%Y-%m-%d %H:%M Asia/Shanghai"),
        "window": "最近24–36小时",
        "items": [
            {
                "id": it.id,
                "lang": it.lang,
                "type": it.type,
                "hotness": it.hotness,
                "title": it.title,
                "summary": it.summary,
                "sources": it.sources,
            }
            for it in top
        ],
    }

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {out_path} with {len(top)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
