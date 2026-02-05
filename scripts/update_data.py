#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Update data.json for GitHub Pages.

Sources:
- Hacker News front page (HTML): capture title+url+points; filter AI-ish.
- 机器之心 RSS: recent AI news (中文)
- The Verge RSS (Atom): include items tagged "AI".
- MIT Technology Review RSS: include items in Artificial intelligence category.

Time window:
- Prefer items published within the last 36 hours.
- Minimum age: at least 24 hours old (i.e., "yesterday-ish"), when timestamp available.
  If a source lacks usable timestamps, keep but down-rank.

Output:
- data.json with:
  - displayTitle: "YYYY年M月D日 AI大事件"
  - generatedAt (Asia/Shanghai)
  - items[]: title (Chinese where possible), summary (Chinese wrapper), sources[]

Note: This script avoids paid APIs / LLM calls. English content is wrapped in Chinese
      where possible; proper nouns remain.
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
    # Avoid pytz dependency; use fixed offset +8
    return datetime.utcnow() + timedelta(hours=8)


def to_shanghai(dt: datetime) -> datetime:
    # If dt is timezone-aware, normalize to UTC then add 8h.
    if dt.tzinfo is not None:
        dt = dt.astimezone(tz=None).replace(tzinfo=None)  # local naive; good enough in GitHub runner
    # parsedate_to_datetime gives aware UTC; easier: convert to UTC naive then +8
    # However we don't have tz database; approximate by dropping tz and assume UTC.
    return dt.replace(tzinfo=None) + timedelta(hours=8)


def within_window(dt: Optional[datetime], *, start: datetime, end: datetime) -> bool:
    if dt is None:
        return False
    return start <= dt <= end


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


def zh_wrap_summary(title: str, lang: str) -> str:
    """Return a short Chinese summary.

    We do *not* call LLMs here. Use lightweight templates + a few hand-coded mappings.
    Proper nouns remain in English.
    """
    if lang == "zh":
        return ""

    t = title.strip()
    tl = t.lower()

    # Small mapping for recurring items
    if "voxtral" in tl and "transcribe" in tl:
        return "Mistral 更新语音转写（speech-to-text）相关能力。用途：会议纪要、客服质检、语音助手等语音→文本场景。"
    if tl.startswith("claude is a space to think") or "space to think" in tl:
        return "Anthropic 解释 Claude 的产品定位：更强调结构化思考与复杂任务推进。用途：规划、写作、分步推理与任务拆解。"
    if "copilot" in tl and "problems" in tl:
        return "媒体报道 Microsoft Copilot 在产品落地/体验上遇到挑战。影响：大厂 AI 产品从“能用”到“规模化好用”仍有鸿沟。"
    if "guardrails" in tl and "governance" in tl and "agent" in tl:
        return "讨论如何用“边界治理”而非纯提示词来管控 agentic systems（权限、工具、数据、审批）。用途：企业上线 AI Agent 的安全清单。"
    if "ai is killing" in tl and "saas" in tl:
        return "观点文章讨论 AI 对传统 B2B SaaS 的冲击（定价、壁垒、产品形态）。用途：提醒 SaaS 团队重估护城河与价值交付方式。"

    return f"英文资讯：{t}。用途/影响：详见来源。"


def parse_hn() -> List[Item]:
    html = fetch("https://news.ycombinator.com/")

    # Extract rows by pairing titleline and subtext score.
    # Title+URL:
    titles = []
    for m in re.finditer(r'<span class="titleline">\s*<a href="(.*?)"[^>]*>(.*?)</a>', html, re.S):
        url = unescape(m.group(1)).strip()
        title = unescape(re.sub(r"<.*?>", "", m.group(2))).strip()
        titles.append((title, url))

    scores = []
    for m in re.finditer(r'<span class="score"[^>]*>(\d+) points</span>', html):
        scores.append(int(m.group(1)))

    items = []
    for idx, (title, url) in enumerate(titles[:30]):
        pts = scores[idx] if idx < len(scores) else 0
        t = title.lower()
        if not any(k in t for k in AI_KEYWORDS):
            continue

        item_id = f"hn-{idx}-{re.sub(r'[^a-z0-9]+','-',t)[:40]}".strip("-")
        itype = classify_type(title)
        items.append(
            Item(
                id=item_id,
                lang="en",
                type=itype,
                hotness=min(100, 40 + pts // 10),
                title=title,
                summary=zh_wrap_summary(title, "en"),
                sources=[
                    {"name": "Hacker News", "url": f"https://news.ycombinator.com/"},
                    {"name": "原文", "url": url},
                ],
                published_at=None,
            )
        )

    return items


def parse_rss_items(xml: str) -> List[Tuple[str, str, Optional[datetime]]]:
    out = []
    for m in re.finditer(r"<item>.*?<title>(.*?)</title>.*?<link>(.*?)</link>.*?<pubDate>(.*?)</pubDate>", xml, re.S):
        title = unescape(re.sub(r"<.*?>", "", m.group(1))).strip()
        link = unescape(m.group(2)).strip()
        pub_raw = m.group(3).strip()
        dt = None
        try:
            dt = parsedate_to_datetime(pub_raw)
            # parsedate_to_datetime is usually aware; convert to UTC-naive then +8
            dt = dt.astimezone(tz=None).replace(tzinfo=None)
        except Exception:
            dt = None
        out.append((title, link, dt))
    return out


def parse_atom_entries(xml: str) -> List[Tuple[str, str, Optional[datetime], List[str]]]:
    out = []
    # For The Verge, title is CDATA and link rel=alternate.
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
        dt = None
        try:
            # ISO 8601
            dt = datetime.fromisoformat(published_raw.replace("Z", "+00:00")).astimezone(tz=None).replace(tzinfo=None)
        except Exception:
            dt = None
        out.append((title, link, dt, cats))
    return out


def parse_jiqizhixin(start: datetime, end: datetime) -> List[Item]:
    xml = fetch("https://www.jiqizhixin.com/rss")
    rows = parse_rss_items(xml)
    items = []
    for i, (title, link, dt_utc_naive) in enumerate(rows):
        dt_sh = None
        if dt_utc_naive is not None:
            # pubDate already includes +0800; parsed then localized to system; treat as local naive
            dt_sh = dt_utc_naive
        # Keep only within end window if possible
        if dt_sh is not None and not within_window(dt_sh, start=start, end=end):
            continue

        itype = classify_type(title)
        items.append(
            Item(
                id=f"jiqizhixin-{i}",
                lang="zh",
                type=itype,
                hotness=60,
                title=title,
                summary="（中文媒体）详见来源链接。",  # minimal; avoids reprinting paywalled/long content
                sources=[{"name": "机器之心", "url": link}],
                published_at=dt_sh,
            )
        )
    return items


def parse_verge(start: datetime, end: datetime) -> List[Item]:
    xml = fetch("https://www.theverge.com/rss/index.xml")
    entries = parse_atom_entries(xml)
    items = []
    for i, (title, link, dt_utc_naive, cats) in enumerate(entries):
        if "AI" not in cats and "Artificial Intelligence" not in cats:
            # allow if title contains AI-ish keyword
            if not any(k in title.lower() for k in AI_KEYWORDS):
                continue
        dt_sh = None
        if dt_utc_naive is not None:
            dt_sh = dt_utc_naive + timedelta(hours=8)  # published is -05 etc; fromisoformat -> local; approximate
        if dt_sh is not None and not within_window(dt_sh, start=start, end=end):
            continue

        items.append(
            Item(
                id=f"verge-{i}",
                lang="en",
                type=classify_type(title),
                hotness=55,
                title=title,
                summary=zh_wrap_summary(title, "en"),
                sources=[{"name": "The Verge", "url": link}],
                published_at=dt_sh,
            )
        )
    return items


def parse_mit_tr(start: datetime, end: datetime) -> List[Item]:
    xml = fetch("https://www.technologyreview.com/feed/")
    rows = parse_rss_items(xml)
    items = []
    for i, (title, link, dt_utc_naive) in enumerate(rows):
        # quick category filter by title keywords only (RSS categories exist but keep simple)
        if not any(k in title.lower() for k in AI_KEYWORDS + ["agentic"]):
            continue
        dt_sh = None
        if dt_utc_naive is not None:
            dt_sh = dt_utc_naive + timedelta(hours=8)
        if dt_sh is not None and not within_window(dt_sh, start=start, end=end):
            continue
        items.append(
            Item(
                id=f"mittr-{i}",
                lang="en",
                type="policy",
                hotness=50,
                title=title,
                summary=zh_wrap_summary(title, "en"),
                sources=[{"name": "MIT Technology Review", "url": link}],
                published_at=dt_sh,
            )
        )
    return items


def dedupe(items: List[Item]) -> List[Item]:
    seen = set()
    out = []
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

    # Filter by min_age when we have timestamps; otherwise keep but downrank.
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

    out_path = os.path.join(os.path.dirname(__file__), "..", "data.json")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {out_path} with {len(top)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
