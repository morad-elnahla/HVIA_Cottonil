"""طبقة التصميم: الألوان والخطوط والكروت والرسومات الصغيرة (HTML/CSS بدون مكتبات إضافية)."""
from __future__ import annotations

import html
import itertools
import re
from urllib.parse import quote

MINT, AMBER, RED, BLUE, GRAY = "#5eeab0", "#f6b94d", "#ff6b6b", "#7aa7ff", "#6f7f79"
STATUS_COLOR = {"حرج": RED, "منخفض": AMBER, "سليم": MINT, "زيادة": BLUE, "راكد": GRAY}
PRI_COLOR = {"حرجة": RED, "مهمة": AMBER, "فرصة": MINT, "معلومة": GRAY}
_uid = itertools.count()

NAV_ICONS = [
    "M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z",
    "M3 17l6-6 4 4 8-8M15 7h6v6",
    "M3 9l1-5h16l1 5M4 9v11h16V9M9 20v-6h6v6",
    "M21 8l-9-5-9 5v8l9 5 9-5zM3 8l9 5 9-5M12 13v9",
    "M12 3l1.8 4.7 4.7 1.8-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8zM19 15l.8 2.2 2.2.8-2.2.8L19 21l-.8-2.2-2.2-.8 2.2-.8z",
    "M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8M22 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8",
    "M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.2 1 2V17h6v-.3c0-.8.4-1.5 1-2A7 7 0 0 0 12 2z",
    "M12 8V4H8M4 12a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2zM2 14h2M20 14h2M9 14v1M15 14v1",
    "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10zM9 12l2 2 4-4",
]


def _icon(path: str) -> str:
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='{path}'/></svg>"
    return f'url("data:image/svg+xml,{quote(svg)}")'


def css(rec_count: int = 0) -> str:
    nav = "".join(f'[data-testid="stSidebar"] div[role="radiogroup"] > div:nth-child({i + 1}) label::before{{-webkit-mask-image:{_icon(p)};mask-image:{_icon(p)};}}' for i, p in enumerate(NAV_ICONS))
    badge = (f'[data-testid="stSidebar"] div[role="radiogroup"] > div:nth-child(7) label::after{{content:"{rec_count}";background:rgba(255,107,107,.16);color:var(--red);border-radius:10px;padding:1px 9px;font:600 12px var(--n);margin-right:auto;}}' if rec_count else "")
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{{--bg:#070b0a;--card:#0d1412;--card2:#111b18;--line:#1b2723;--tx:#eef3f1;--mut:#8b9b95;--mint:{MINT};--amb:{AMBER};--red:{RED};--blue:{BLUE};
--f:'IBM Plex Sans Arabic','Segoe UI',Tahoma,sans-serif;--n:'Space Grotesk','IBM Plex Sans Arabic','Segoe UI',sans-serif;}}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stSidebar"]{{font-family:var(--f)!important;direction:rtl;text-align:right;color:var(--tx);}}
.stApp{{background:radial-gradient(1100px 520px at 20% -12%,rgba(94,234,176,.10),transparent 60%),radial-gradient(900px 420px at 92% -10%,rgba(139,92,246,.10),transparent 60%),var(--bg);}}
.stApp::before{{content:"";position:fixed;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#6d4be0,#5eeab0 50%,#6d4be0);opacity:.75;z-index:1000;}}
[data-testid="stToolbar"],[data-testid="stDecoration"],footer{{display:none!important;}}
[data-testid="stHeader"]{{background:transparent!important;height:0!important;}}
@media (min-width:901px){{[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapsedControl"]{{display:none!important;}}}}
@media (max-width:900px){{[data-testid="stSidebarCollapsedControl"],[data-testid="stSidebarCollapseButton"]{{display:flex!important;}} [data-testid="stHeader"]{{height:2.6rem!important;}}}}
.block-container{{padding:1.4rem 2rem 3rem!important;max-width:1500px;}}
h1,h2,h3,h4,p,li,label,span,div{{font-family:var(--f);}}
/* الشريط الجانبي */
[data-testid="stSidebar"]{{background:linear-gradient(180deg,#0a100e,#080c0b);border-left:1px solid var(--line);border-right:none;min-width:280px!important;max-width:280px!important;}}
[data-testid="stSidebar"] > div{{padding:1.2rem .9rem;}}
.brand{{display:flex;align-items:center;gap:12px;padding:6px 8px 22px;}}
.brand .logo{{width:46px;height:46px;border-radius:50%;background:var(--mint);color:#04120c;display:flex;align-items:center;justify-content:center;font:700 20px var(--n);}}
.brand b{{display:block;font:700 19px var(--n);letter-spacing:.5px;}} .brand small{{color:var(--mut);font-size:12.5px;}}
[data-testid="stSidebar"] [data-testid="stRadio"] > label{{display:none;}}
[data-testid="stSidebar"] div[role="radiogroup"]{{gap:4px;}}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]{{display:flex;flex-direction:row;align-items:center;gap:12px;padding:12px 16px;border-radius:14px;cursor:pointer;color:#c5d0cb;width:100%;transition:background .15s;}}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div{{display:flex;align-items:center;flex:1;margin:0;}}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child{{display:none;}}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] p{{font-size:15.5px;font-weight:500;margin:0;color:inherit;}}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]::before{{content:"";width:20px;height:20px;background:currentColor;-webkit-mask-size:contain;mask-size:contain;-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;flex:none;}}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover{{background:rgba(255,255,255,.04);}}
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"]{{background:rgba(94,234,176,.10);color:var(--mint);}}
{badge}
{nav}
.fresh{{margin-top:30vh;background:var(--card);border:1px solid var(--line);border-radius:18px;padding:14px 16px;font-size:13px;color:var(--mut);}}
.fresh b{{color:var(--mint);font-weight:600;display:flex;align-items:center;gap:8px;}} .fresh b::before{{content:"";width:8px;height:8px;border-radius:50%;background:var(--mint);box-shadow:0 0 10px var(--mint);}}
.fresh.warn b{{color:var(--amb);}} .fresh.warn b::before{{background:var(--amb);box-shadow:0 0 10px var(--amb);}}
/* الكروت */
.card,[class*="st-key-card"]{{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:22px;padding:22px 24px;}}
[class*="st-key-card"]{{margin-bottom:6px;}}
.sect{{margin-bottom:14px;}} .ptitle,.headline{{padding:0!important;}} .sect-t{{font-size:19px;font-weight:700;line-height:1.5;text-align:right;}} .sect p{{margin:3px 0 0;color:var(--mut);font-size:13.5px;}}
.hello{{color:var(--mut);font-size:14.5px;margin-bottom:6px;}}
.headline{{font-size:40px;font-weight:700;line-height:1.25;margin:0 0 10px;}} .headline em{{font-style:normal;color:var(--mint);}}
.ptitle{{font-size:34px;font-weight:700;margin:0 0 4px;}} .psub{{color:var(--mut);font-size:15px;margin-bottom:18px;}}
.kpi{{position:relative;overflow:hidden;background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:22px;padding:18px 22px 0;height:176px;}}
.kpi .top{{display:flex;justify-content:space-between;align-items:center;}} .kpi .lab{{color:#b7c4be;font-size:15px;}}
.kpi .val{{font:700 44px/1.15 var(--n);margin-top:8px;display:flex;align-items:baseline;gap:8px;}} .kpi .val small{{font:500 15px var(--f);color:var(--mut);}}
.kpi svg{{position:absolute;bottom:0;left:0;width:100%;height:58px;}}
.pill{{display:inline-flex;align-items:center;gap:4px;padding:3px 11px;border-radius:12px;font:600 12.5px var(--n);direction:ltr;}}
.pill.up{{background:rgba(94,234,176,.13);color:var(--mint);}} .pill.down{{background:rgba(255,107,107,.14);color:var(--red);}} .pill.flat{{background:rgba(255,255,255,.07);color:var(--mut);}}
.tag{{display:inline-block;padding:3px 12px;border-radius:12px;font-size:12.5px;font-weight:600;}}
.pc{{height:100%;}} .pc .hd{{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;}}
.pc .rank{{width:30px;height:30px;border-radius:10px;background:rgba(255,255,255,.07);display:flex;align-items:center;justify-content:center;font:600 14px var(--n);}}
.pc-t{{margin:0 0 6px;font-size:19px;font-weight:700;line-height:1.5;min-height:58px;text-align:right;}} .pc .big{{font:700 38px var(--n);margin:4px 0 0;}} .pc .big small{{font:500 14px var(--f);color:var(--mut);margin-right:6px;}}
.pc .lbl{{color:var(--mut);font-size:13.5px;margin-bottom:14px;}} .pc .ev{{color:#b7c4be;font-size:14px;line-height:1.75;margin-top:14px;}}
.ring{{width:92px;height:92px;border-radius:50%;display:flex;align-items:center;justify-content:center;font:700 22px var(--n);}} .ring span{{background:var(--card);border-radius:50%;width:70px;height:70px;display:flex;align-items:center;justify-content:center;}}
.rk-row{{display:grid;grid-template-columns:30% 1fr 150px;gap:10px;align-items:center;margin:11px 0;font-size:14px;}}
.rk-l{{color:#cdd8d3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}} .rk-t{{height:12px;background:rgba(255,255,255,.06);border-radius:8px;overflow:hidden;direction:rtl;}} .rk-t i{{display:block;height:100%;border-radius:8px;}}
.rk-v{{color:var(--mut);font:500 12.5px var(--n);text-align:left;white-space:nowrap;}}
.dv-row{{display:grid;grid-template-columns:30% 1fr 130px;gap:10px;align-items:center;margin:9px 0;font-size:14px;}}
.dv-t{{position:relative;height:14px;background:rgba(255,255,255,.05);border-radius:8px;}} .dv-t::after{{content:"";position:absolute;right:50%;top:-3px;bottom:-3px;width:1px;background:#33423d;}}
.dv-t i{{position:absolute;top:0;height:100%;border-radius:6px;}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:16px 18px;}} .stat small{{color:var(--mut);font-size:13px;display:block;}} .stat b{{font:700 28px var(--n);display:block;margin:4px 0 2px;}} .stat span{{color:var(--mut);font-size:12.5px;}}
.note{{background:rgba(94,234,176,.06);border:1px solid rgba(94,234,176,.22);border-radius:16px;padding:12px 16px;color:#cfe9de;font-size:14px;line-height:1.8;margin:10px 0;}}
.note.warn{{background:rgba(246,185,77,.07);border-color:rgba(246,185,77,.28);color:#f1dfb9;}} .note.bad{{background:rgba(255,107,107,.07);border-color:rgba(255,107,107,.3);color:#f6cccc;}}
.cap{{color:var(--mut);font-size:13px;margin:8px 2px 0;line-height:1.7;}} .cap b{{color:#cdd8d3;font-weight:600;}}
.tbl{{width:100%;border-collapse:collapse;font-size:14px;}} .tbl th{{color:var(--mut);font-weight:500;font-size:12.5px;text-align:right;padding:8px 10px;border-bottom:1px solid var(--line);}}
.tbl td{{padding:11px 10px;border-bottom:1px solid rgba(255,255,255,.045);}} .tbl tr:last-child td{{border-bottom:none;}} .tbl .n{{font-family:var(--n);direction:ltr;text-align:left;}}
.cov-h,.cov-r{{display:grid;grid-template-columns:1.5fr .8fr .8fr 2.4fr .8fr 1.3fr;gap:12px;align-items:center;}}
.cov-h{{color:var(--mut);font-size:12.5px;padding:0 6px 8px;border-bottom:1px solid var(--line);}} .cov-r{{padding:13px 6px;border-bottom:1px solid rgba(255,255,255,.045);font-size:14px;}}
.cov-r:last-child{{border-bottom:none;}} .cov-n{{font-family:var(--n);direction:ltr;text-align:right;}} .cov-r small{{display:block;color:var(--mut);font-size:12px;}}
.cov-b{{position:relative;height:16px;background:rgba(255,255,255,.05);border-radius:9px;direction:rtl;}} .cov-z{{position:absolute;top:0;bottom:0;background:rgba(94,234,176,.16);border-left:1px dashed rgba(94,234,176,.55);border-right:1px dashed rgba(94,234,176,.55);}}
.cov-f{{position:absolute;top:3px;bottom:3px;right:0;border-radius:8px;}} .cov-t{{position:absolute;top:-1px;font:600 11.5px var(--n);color:#fff;mix-blend-mode:difference;}}
.cov-ax{{display:flex;justify-content:space-between;font:500 11px var(--n);color:var(--mut);direction:rtl;margin-top:3px;}}
.qr{{display:flex;gap:14px;align-items:center;padding:14px 0;border-bottom:1px solid rgba(255,255,255,.05);}} .qr:last-child{{border-bottom:none;}}
.qr .ic{{width:40px;height:40px;border-radius:12px;display:flex;align-items:center;justify-content:center;flex:none;font-size:18px;}} .qr b{{display:block;font-size:15px;font-weight:600;}} .qr span{{color:var(--mut);font-size:13px;}}
.sec-b{{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0;}} .src{{color:var(--mut);font-size:12.5px;border-top:1px dashed var(--line);margin-top:12px;padding-top:8px;}}
.lbl-fact,.lbl-fc,.lbl-inf{{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:600;margin-left:8px;}}
.lbl-fact{{background:rgba(94,234,176,.14);color:var(--mint);}} .lbl-fc{{background:rgba(246,185,77,.14);color:var(--amb);}} .lbl-inf{{background:rgba(122,167,255,.14);color:var(--blue);}}
/* عناصر Streamlit الأصلية */
div[data-baseweb="select"] > div,div[data-baseweb="input"],div[data-baseweb="base-input"],[data-testid="stTextInput"] input{{background:var(--card)!important;border:1px solid var(--line)!important;border-radius:14px!important;}}
[data-testid="stPopover"] > div > button,.stButton > button,.stDownloadButton > button{{background:var(--card)!important;border:1px solid var(--line)!important;border-radius:14px!important;color:var(--tx)!important;font-family:var(--f)!important;}}
.stButton > button:hover,.stDownloadButton > button:hover,[data-testid="stPopover"] > div > button:hover{{border-color:var(--mint)!important;color:var(--mint)!important;}}
.stButton > button[kind="primary"]{{background:var(--mint)!important;color:#04120c!important;border-color:var(--mint)!important;font-weight:700;}}
[class*="st-key-link"] button{{background:transparent!important;border:none!important;color:var(--mint)!important;padding:0!important;font-weight:600;justify-content:flex-start;}}
button[data-baseweb="tab"]{{font-family:var(--f);font-size:15px;}} [data-baseweb="tab-highlight"]{{background:var(--mint)!important;}}
[data-testid="stExpander"]{{border:1px solid var(--line)!important;border-radius:16px!important;background:var(--card);}}
[data-testid="stMetric"]{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:14px 18px;}}
[data-testid="stChatInput"]{{border-radius:18px;}} [data-testid="stDataFrame"]{{border-radius:14px;overflow:hidden;}}
@media (max-width:900px){{.headline{{font-size:28px;}}}}

/* مساواة ارتفاع الكاردز في نفس الصف */
[data-testid="stHorizontalBlock"]{{align-items:stretch!important;}}
[data-testid="stColumn"]{{display:flex!important;flex-direction:column;}}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"]{{flex:1;}}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> [class*="st-key-card"]){{flex:1;display:flex;flex-direction:column;}}
[data-testid="stLayoutWrapper"] > [class*="st-key-card"]{{flex:1;}}
.stat{{min-height:112px;height:100%;}} .stat small,.stat span{{line-height:1.6;}}
/* القوائم والتاجات: خلفية غامقة ونص واضح */
span[data-tag],span[data-baseweb="tag"]{{background:rgba(94,234,176,.14)!important;color:#d8f7ea!important;border:1px solid rgba(94,234,176,.42)!important;border-radius:10px!important;}}
span[data-tag] *,span[data-baseweb="tag"] *{{color:#d8f7ea!important;fill:#d8f7ea!important;}}
span[data-tag]:hover{{background:rgba(94,234,176,.24)!important;}}
div[data-baseweb="popover"] > div,[data-baseweb="menu"],ul[role="listbox"]{{background:#0f1815!important;color:#eef3f1!important;border:1px solid #243530!important;border-radius:14px!important;}}
[data-baseweb="menu"] li,ul[role="listbox"] li,[role="option"]{{background:transparent!important;color:#eef3f1!important;text-align:right;}}
[data-baseweb="menu"] li:hover,ul[role="listbox"] li:hover,[role="option"]:hover,[role="option"][aria-selected="true"]{{background:rgba(94,234,176,.14)!important;}}
[data-testid="stSelectbox"] [data-baseweb="select"]{{background:var(--card)!important;border:1px solid var(--line)!important;border-radius:14px!important;}}
[data-baseweb="select"] > div{{border:none!important;}} [data-baseweb="select"] input{{color:#eef3f1!important;}}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"],[data-testid="stSlider"] [data-testid="stSliderThumbValue"] *{{color:#eef3f1!important;background:transparent!important;}}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{{background:var(--mint)!important;}}
[data-testid="stFileUploader"] section{{background:var(--card)!important;border:1px dashed #2c3f39!important;border-radius:16px!important;color:#cdd8d3!important;}}
[data-testid="stFileUploader"] *{{color:#cdd8d3;}}
[data-testid="stStatusWidget"],[data-testid="stStatus"]{{background:var(--card)!important;border:1px solid var(--line)!important;border-radius:16px!important;}}
button[kind="pills"]{{background:var(--card)!important;border:1px solid var(--line)!important;color:#c5d0cb!important;border-radius:16px!important;font-family:var(--f)!important;}}
button[kind="pillsActive"]{{background:rgba(94,234,176,.16)!important;border:1px solid rgba(94,234,176,.55)!important;color:#d8f7ea!important;border-radius:16px!important;font-family:var(--f)!important;}}
[data-testid="stPills"]{{direction:rtl;}}
[data-testid="stPills"] [data-testid="stButtonGroup"],[data-testid="stPills"] [role="group"]{{display:flex;flex-wrap:wrap;gap:8px;}}
button[kind="pills"],button[kind="pillsActive"]{{min-height:38px;max-width:100%;padding:6px 14px;white-space:normal;}}
button[kind="pills"]:hover{{border-color:var(--mint)!important;}}
button[kind="pills"]:focus-visible,button[kind="pillsActive"]:focus-visible{{outline:2px solid var(--mint);outline-offset:2px;}}
.kind{{display:inline-block;padding:2px 9px;border-radius:9px;font-size:11.5px;font-weight:600;margin-right:6px;border:1px solid;}}
.kind.k-حقيقة{{color:var(--mint);border-color:rgba(94,234,176,.4);}} .kind.k-تقدير{{color:var(--amb);border-color:rgba(246,185,77,.45);}}
.kind.k-توقع{{color:var(--blue);border-color:rgba(122,167,255,.45);}} .kind.k-استنتاج{{color:#c9a7ff;border-color:rgba(201,167,255,.45);}}
.tw,.cov-wrap{{width:100%;max-width:100%;min-width:0;overflow-x:auto;direction:rtl;overscroll-behavior-x:contain;scrollbar-color:#365448 var(--card);}}
.cov-wrap > div{{min-width:800px;}}
.cov-h > *,.cov-r > *{{min-width:0;overflow-wrap:anywhere;}}
.tbl{{direction:rtl;}} .tbl th,.tbl td{{vertical-align:top;text-align:right;}}
.tbl .n{{white-space:nowrap;unicode-bidi:isolate;text-align:left;}}
[data-testid="stColumn"],[data-testid="stVerticalBlock"],[data-testid="stLayoutWrapper"],[data-testid="stMarkdownContainer"]{{min-width:0;}}
.card,[class*="st-key-card"],.stat{{box-sizing:border-box;overflow-wrap:anywhere;}}
.chip-row{{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 2px;}} .chip{{padding:3px 12px;border-radius:12px;font-size:12.5px;font-weight:600;background:rgba(255,255,255,.06);color:#c5d0cb;}}
.chip.ai{{background:rgba(94,234,176,.14);color:var(--mint);}} .chip.warn{{background:rgba(246,185,77,.14);color:var(--amb);}}
.sumbox{{background:rgba(94,234,176,.07);border:1px solid rgba(94,234,176,.3);border-radius:16px;padding:12px 16px;margin:6px 0 10px;line-height:1.9;}}
.part{{border-top:1px solid var(--line);margin-top:14px;padding-top:12px;}} .part-t{{font-weight:700;color:var(--mint);margin-bottom:6px;}}
/* اتجاه النص مستقل عن محاذاة الفقاعات والقيم الرقمية */
.tbl th.num-head{{text-align:left;direction:rtl;}}
.tbl td:not(.n){{direction:rtl;unicode-bidi:plaintext;}}
.tbl .n,.cov-n{{direction:ltr;unicode-bidi:isolate;font-variant-numeric:tabular-nums;}}
.answer-list{{direction:rtl;text-align:right;list-style-position:outside;padding-inline-start:1.25rem;margin:4px 0;color:#cdd8d3;font-size:14.5px;line-height:1.85;}}
.answer-list li{{padding-inline-start:.2rem;margin:4px 0;}}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"]{{direction:rtl;text-align:right!important;font-size:16px;line-height:1.85;}}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] p,[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] li{{direction:rtl;text-align:right!important;unicode-bidi:normal;font-size:16px;}}
[data-testid="stChatMessageContent"] .answer-list{{padding-inline-start:1.4rem;padding-inline-end:0;margin:6px 0;}}
[data-testid="stChatMessageContent"] .answer-list li::marker{{color:var(--mint);}}
[data-testid="stChatMessageContent"] .cap,[data-testid="stChatMessageContent"] .src{{font-size:14px;line-height:1.85;}}
[data-testid="stChatMessageContent"] .lbl-fact,[data-testid="stChatMessageContent"] .lbl-fc,[data-testid="stChatMessageContent"] .lbl-inf,[data-testid="stChatMessageContent"] .chip{{font-size:13px;}}
[data-testid="stChatMessageContent"] .tbl{{font-size:15px;}}
[data-testid="stChatMessageContent"] .tbl th{{font-size:14px;line-height:1.6;min-width:100px;}}
[data-testid="stChatMessageContent"] .tbl th.num-head,[data-testid="stChatMessageContent"] .tbl td.n{{text-align:left!important;}}
[data-testid="stChatMessageContent"] .tbl td.n{{direction:ltr;unicode-bidi:isolate;}}
[data-testid="stChatMessageContent"] .tbl td:not(.n),[data-testid="stChatMessageContent"] .tbl th:not(.num-head){{text-align:right!important;}}
[data-testid="stChatMessageContent"] pre,[data-testid="stChatMessageContent"] code{{direction:ltr;unicode-bidi:isolate;text-align:left;}}
[data-baseweb="select"] input,[data-testid="stTextInput"] input{{text-align:start;unicode-bidi:plaintext;}}
[data-baseweb="select"] [data-baseweb="tag"],[role="option"]{{direction:rtl;unicode-bidi:plaintext;}}
[data-baseweb="select"] svg,.qr .ic{{flex-shrink:0;}}
.kpi .val,.pc .big,.stat b{{font-variant-numeric:tabular-nums;}}
/* محاذاة صريحة لا تعتمد على اتجاه flex الموروث */
[data-testid="stChatMessage"]{{display:flex!important;flex-direction:row!important;direction:ltr;width:100%!important;max-width:100%!important;box-sizing:border-box;margin:0!important;gap:0;padding:6px 0!important;background:transparent!important;}}
[data-testid="stChatMessage"]:has(.who-user){{justify-content:flex-end;}}
[data-testid="stChatMessage"]:has(.who-bot){{justify-content:flex-start;}}
[data-testid="stChatMessage"] > [aria-hidden="true"],[data-testid="stChatMessageAvatarUser"],[data-testid="stChatMessageAvatarAssistant"]{{display:none!important;}}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"]{{direction:rtl;text-align:right;min-width:0;width:fit-content;flex:0 1 auto;box-sizing:border-box;overflow-wrap:anywhere;color:var(--tx);}}
[data-testid="stChatMessage"]:has(.who-user) [data-testid="stChatMessageContent"]{{background:#12392f;border:1px solid #28654f;border-radius:18px 18px 4px 18px;padding:10px 16px;max-width:75%;margin-left:auto!important;margin-right:0!important;}}
[data-testid="stChatMessage"]:has(.who-bot) [data-testid="stChatMessageContent"]{{background:var(--card2);border:1px solid #26382f;border-radius:18px 18px 18px 4px;padding:14px 18px;max-width:90%;margin-right:auto!important;margin-left:0!important;}}
.who-user,.who-bot{{display:none;}}
[data-testid="stChatMessageContent"] [data-testid="stElementContainer"]:has(.who-user),[data-testid="stChatMessageContent"] [data-testid="stElementContainer"]:has(.who-bot){{display:none;}}
[data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"]{{gap:.45rem;}}
[data-testid="stChatMessageContent"] p{{margin:0;line-height:1.8;}}
[data-testid="stChatInput"] textarea{{direction:rtl;text-align:right;}}
.st-key-chat_suggestions [data-testid="stHorizontalBlock"]{{gap:.5rem;}}
.st-key-chat_suggestions button{{height:100%;min-height:44px;text-align:right;padding:8px 12px;}}
.st-key-chat_suggestions button p{{font-size:13px;line-height:1.6;}}
@media (max-width:600px){{
.card,[class*="st-key-card"]{{padding:16px 14px;border-radius:16px;}}
.st-key-chat_suggestions [data-testid="stColumn"]{{min-width:0!important;flex:1 1 calc(50% - .5rem)!important;width:calc(50% - .5rem)!important;}}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"]{{padding:10px 12px!important;}}
.kpi{{padding-inline:16px;}} .kpi .val{{font-size:30px;}} .pc .big{{font-size:30px;}}
}}
@media (max-width:900px){{.rk-row,.dv-row{{grid-template-columns:36% 1fr;}}.rk-v{{grid-column:1/-1;text-align:right;margin-top:-4px;}}.block-container{{padding:1rem .8rem 2rem!important;}}.headline{{font-size:26px;}}.ptitle{{font-size:26px;}}.kpi .val{{font-size:34px;}}.pc-t{{min-height:0;}}
[data-testid="stSidebar"]{{min-width:220px!important;max-width:220px!important;}}[data-testid="stChatMessageContent"]{{max-width:92%!important;}}}}
</style>"""


# ---------------------------------------------------------------- مكونات HTML
_NUM = re.compile(r"[A-Za-z][A-Za-z0-9_./:@+%\-]*(?:[ ]+[A-Za-z][A-Za-z0-9_./:@+%\-]*)*|\d{4}-\d{2}(?:-\d{2})?(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?(?!\d)|\d+(?:-\d+)+|\d{1,2}:\d{2}(?::\d{2})?|[+\-−]?\d[\d,]*(?:\.\d+)?%?")
LRI, PDI = "\u2066", "\u2069"


def bidi(s) -> str:
    """بيعزل الأرقام والتواريخ والنسب (LRI/PDI) عشان تتعرض بالترتيب الصح جوه الجمل العربي (مثلًا +30% و2026-10-18 و09:55)."""
    # Idempotent: H() may receive text already escaped by esc().
    parts = re.split(r"(\u2066[^\u2069]*\u2069)", str(s))
    return "".join(p if p.startswith(LRI) else _NUM.sub(lambda m: f"{LRI}{m.group(0)}{PDI}", p) for p in parts)


def bidi_html(h: str) -> str:
    """نفس الفكرة على نص HTML: بنعزل الأرقام اللي بره التاجات بس."""
    return "".join(p if p.startswith(("<", "&")) else bidi(p) for p in re.split(r"(<[^>]+>|&(?:#[0-9]+|#x[0-9a-fA-F]+|[A-Za-z]+);)", h))


def esc(s) -> str:
    return html.escape(bidi(s))


def money_parts(v: float) -> tuple[str, str]:
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:,.2f}", "مليار ج.م"
    if a >= 1e6:
        return f"{v / 1e6:,.1f}", "مليون ج.م"
    if a >= 1e4:
        return f"{v / 1e3:,.0f}", "ألف ج.م"
    return f"{v:,.0f}", "ج.م"


def sparkline(vals, color=MINT, w=300, h=58) -> str:
    vals = [float(v) for v in vals if v == v]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    pts = [(i * w / (len(vals) - 1), h - 8 - (v - lo) / rng * (h - 22)) for i, v in enumerate(vals)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    gid = f"g{next(_uid)}"
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{color}" stop-opacity=".30"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs><polygon points="0,{h} {line} {w},{h}" fill="url(#{gid})"/>'
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.2" vector-effect="non-scaling-stroke"/></svg>')


def pill(text: str, kind: str = "flat", arrow: str | None = None) -> str:
    arrow = arrow if arrow is not None else {"up": "↗", "down": "↘", "flat": ""}[kind]
    return f'<span class="pill {kind}">{esc(text)} {arrow}</span>'


def delta_pill(cur: float, prev: float, inverse: bool = False, points: bool = False) -> str:
    if prev in (0, None) and not points:
        return pill("جديد", "flat")
    d = cur - prev if points else (cur / prev - 1) * 100
    if abs(d) < 0.05:
        return pill("ثابت", "flat")
    good = (d > 0) != inverse
    txt = f"{d:+.1f} نقطة" if points else f"{d:+.1f}%"
    return pill(txt, "up" if good else "down", "↗" if d > 0 else "↘")


def kpi_card(label: str, value: str, unit: str, delta_html: str, series, color=MINT) -> str:
    return (f'<div class="kpi"><div class="top"><span class="lab">{esc(label)}</span>{delta_html}</div>'
            f'<div class="val">{esc(value)}<small>{esc(unit)}</small></div>{sparkline(series, color)}</div>')


def sect(title: str, sub: str = "") -> str:
    return f'<div class="sect"><div class="sect-t">{esc(title)}</div>' + (f"<p>{esc(sub)}</p>" if sub else "") + "</div>"


def tag(text: str, color: str) -> str:
    fg = "#a9b9b3" if color == GRAY else color
    return f'<span class="tag" style="background:{color}24;color:{fg}">{esc(text)}</span>'


def note(text: str, kind: str = "") -> str:
    return f'<div class="note {kind}">{text}</div>'


def stat(label: str, value: str, sub: str = "", color: str | None = None) -> str:
    st = f' style="color:{color}"' if color else ""
    return f'<div class="stat"><small>{esc(label)}</small><b{st}>{esc(value)}</b><span>{esc(sub)}</span></div>'


def ranked(items, color=MINT, highlight_first=True) -> str:
    """items = [(label, number, text)]: أعمدة رفيعة مرتبة، الأطول = الأكبر."""
    if not items:
        return ""
    mx = max(abs(v) for _, v, _ in items) or 1
    rows = []
    for i, (lab, v, txt) in enumerate(items):
        c = color if (i == 0 or not highlight_first) else "#2f6f58" if color == MINT else color
        rows.append(f'<div class="rk-row"><span class="rk-l">{esc(lab)}</span><div class="rk-t"><i style="width:{abs(v) / mx * 100:.1f}%;background:{c}"></i></div><span class="rk-v">{esc(txt)}</span></div>')
    return "".join(rows)


def diverge(items) -> str:
    """items = [(label, signed_number, text)]: أخضر = زيادة، أحمر = نقص، حول خط الصفر."""
    if not items:
        return ""
    mx = max(abs(v) for _, v, _ in items) or 1
    rows = []
    for lab, v, txt in items:
        w = abs(v) / mx * 50
        side = f"right:50%;width:{w:.1f}%;background:{MINT}" if v >= 0 else f"left:50%;width:{w:.1f}%;background:{RED}"
        rows.append(f'<div class="dv-row"><span class="rk-l">{esc(lab)}</span><div class="dv-t"><i style="{side}"></i></div><span class="rk-v" style="color:{MINT if v >= 0 else RED}">{esc(txt)}</span></div>')
    return "".join(rows)


def ring(pct: float, color=AMBER) -> str:
    return f'<div class="ring" style="background:conic-gradient({color} {pct * 3.6:.0f}deg,rgba(255,255,255,.08) 0)"><span>{pct:.0f}%</span></div>'


def problem_card(rank: int, priority: str, title: str, big: str, big_unit: str, label: str, items, evidence: str, ring_pct: float | None = None, kind: str = "") -> str:
    col = PRI_COLOR.get(priority, GRAY)
    viz = ring(ring_pct, col) if (ring_pct is not None and ring_pct == ring_pct) else ranked(items, col)
    return (f'<div class="pc"><div class="hd"><span class="rank">{rank}</span><span>{tag(priority, col)}' + (f'<span class="kind k-{esc(kind)}">{esc(kind)}</span>' if kind else "") + f'</span></div><div class="pc-t">{esc(title)}</div>'
            f'<div class="big">{esc(big)}<small>{esc(big_unit)}</small></div><div class="lbl">{esc(label)}</div>{viz}<div class="ev">{esc(evidence)}</div></div>')


def quick_row(icon: str, color: str, title: str, sub: str) -> str:
    return f'<div class="qr"><span class="ic" style="background:{color}22;color:{color}">{icon}</span><div><b>{esc(title)}</b><span>{esc(sub)}</span></div></div>'


def html_table(headers, rows, numeric=()) -> str:
    th = "".join(f'<th scope="col" class="{"num-head" if i in numeric else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"n" if i in numeric else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tw" dir="rtl"><table class="tbl"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def dataframe_html(df, max_rows=None) -> str:
    """Escaped RTL table; keep money precision and never silently truncate answers."""
    import numbers
    import pandas as pd
    d = df if max_rows is None else df.head(max_rows)
    heads, rows = list(d.columns), []
    for row in d.itertuples(index=False, name=None):
        cells = []
        for column, value in zip(heads, row):
            if pd.isna(value):
                label = "غير متاح"
            elif isinstance(value, numbers.Real) and not isinstance(value, bool):
                if "%" in str(column):
                    label = f"{value:,.2f}%"
                elif float(value).is_integer():
                    label = f"{value:,.0f}"
                else:
                    label = f"{value:,.2f}"
            else:
                label = str(value)
            cells.append(esc(label))
        rows.append(cells)
    result = html_table(heads, rows, numeric=[i for i, col in enumerate(heads) if pd.api.types.is_numeric_dtype(d[col])])
    if len(d) < len(df):
        result += f'<div class="cap">{esc(f"معروض {len(d)} من {len(df)} صف")}</div>'
    return result


def cover_bar(weeks: float, status: str, lead_safety: float, over: float, cap: float = 52) -> str:
    """شريط يوضح كام أسبوع المخزون بيكفي. المنطقة الخضراء المتقطعة = المدى السليم."""
    w = min(weeks, cap) / cap * 100 if weeks == weeks else 100
    z0, z1 = lead_safety / cap * 100, min(over, cap) / cap * 100
    c = STATUS_COLOR.get(status, MINT)
    return (f'<div class="cov-b"><div class="cov-z" style="right:{z0:.1f}%;width:{z1 - z0:.1f}%"></div><div class="cov-f" style="width:{max(w, 2):.1f}%;background:{c}"></div></div>')


def coverage_table(rows, lead_safety: float, over: float, first_col="النوع") -> str:
    """rows: dict(name, sub, stock, demand, cover, status, action)"""
    head = (f'<div class="cov-h"><span>{esc(first_col)}</span><span>المخزون (قطعة)</span><span>بيبيع في الأسبوع</span><span>بيكفي كام أسبوع؟ <small>(المنطقة الخضراء = المدى السليم)</small></span><span>الحالة</span><span>القرار</span></div>')
    out = []
    for r in rows:
        c = r["cover"]
        txt = "مفيش مبيعات" if c == float("inf") else (f"{c:.0f}+ أسبوع" if c >= 52 else (f"{c * 7:.0f} يوم" if c < 2 else f"{c:.1f} أسبوع"))
        out.append(f'<div class="cov-r"><div><b>{esc(r["name"])}</b><small>{esc(r.get("sub", ""))}</small></div><div class="cov-n">{r["stock"]:,.0f}</div><div class="cov-n">{r["demand"]:,.1f}</div>'
                   f'<div>{cover_bar(min(c, 999), r["status"], lead_safety, over)}<small style="margin-top:4px">{txt}</small></div><div>{tag(r["status"], STATUS_COLOR.get(r["status"], GRAY))}</div><div style="font-size:13px">{esc(r.get("action", ""))}</div></div>')
    return '<div class="cov-wrap"><div>' + head + "".join(out) + "</div></div>"


def fig_style(fig, h: int = 340, legend: bool = True):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=8, t=10, b=8),
                      font=dict(family="IBM Plex Sans Arabic, Segoe UI, Tahoma", size=13, color="#9fb0aa"), showlegend=legend,
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor="#0d1412", bordercolor="#2a3a35", font=dict(color="#eef3f1", size=13)))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#22302b", tickfont=dict(size=12), tickangle=0, automargin=True)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False, tickfont=dict(size=12), automargin=True)
    return fig
