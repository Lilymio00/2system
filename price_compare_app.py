# -*- coding: utf-8 -*-
"""
MOMO × PChome 比價系統（Streamlit）
執行方式：streamlit run price_compare_app.py

整合 momo_app.py 與 pchome_app.py，並新增：
  1. 兩平台同時爬取（多執行緒平行 + 即時進度條）、結果快取 10 分鐘
  2. 兩平台價格總覽比價（分布、均價/中位數、勝負判定）
  3. 同款商品自動配對（字元 bigram 相似度 + 型號數字比對）
  4. 互動圖表：Plotly 縮放/懸停/範圍滑桿、散佈圖框選配對商品、即時篩選
  5. 完成特效：toast 通知與氣球（可關閉）
"""
import re
import time
import queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from curl_cffi import requests

# ------------------------------------------------------------------
# 基本設定
# ------------------------------------------------------------------
st.set_page_config(page_title="MOMO × PChome 比價系統", page_icon="🛒", layout="wide")

CJK_FONT = (
    "Microsoft JhengHei, PingFang TC, Noto Sans TC, Noto Sans CJK TC, "
    "Heiti TC, WenQuanYi Micro Hei, sans-serif"
)
COLORS = {"MOMO": "#E6007E", "PChome": "#0072CE"}
CACHE_TTL = 600  # 秒

# 相容新舊版 Streamlit 的「滿版寬度」參數
_ver = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
STRETCH = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}
PLOT_CFG = {"displaylogo": False, "scrollZoom": True}

MOMO_API = "https://apisearch.momoshop.com.tw/momoSearchCloud/moec/textSearch"
MOMO_HOME = "https://www.momoshop.com.tw/"
MOMO_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Content-Type": "application/json;charset=UTF-8",
    "Origin": "https://www.momoshop.com.tw",
    "Referer": "https://www.momoshop.com.tw/",
}
PCHOME_API = "https://ecshweb.pchome.com.tw/search/v3.3/all/results"
PCHOME_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Referer": "https://24h.pchome.com.tw/",
}


# ------------------------------------------------------------------
# 爬蟲
# ------------------------------------------------------------------
def parse_price(text) -> float:
    """'$1,299' -> 1299；'1,299~1,599' 取第一個數字；無法解析回傳 NaN"""
    m = re.search(r"\d[\d,]*", str(text))
    return float(m.group().replace(",", "")) if m else float("nan")


def finalize(rows: list, platform: str) -> pd.DataFrame:
    """統一欄位格式：平台 / 序號 / 頁數 / 商品代碼 / 品名 / 價格 / 連結"""
    cols = ["平台", "序號", "頁數", "商品代碼", "品名", "價格", "連結"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["價格"] = pd.to_numeric(df["價格"], errors="coerce")
    df = df.dropna(subset=["價格"])
    df = df[df["價格"] > 0]
    df["價格"] = df["價格"].astype(int)
    df = df.drop_duplicates(subset=["商品代碼", "品名"]).reset_index(drop=True)
    df["平台"] = platform
    df["序號"] = df.index + 1  # 依搜尋結果順序
    return df[cols]


def crawl_momo(keyword: str, pages: int, delay: float, report) -> pd.DataFrame:
    session = requests.Session(impersonate="chrome")  # 模擬 Chrome TLS 指紋
    session.get(MOMO_HOME, headers=MOMO_HEADERS, timeout=30)  # 先取得 cookie

    rows = []
    for page in range(1, pages + 1):
        payload = {
            "host": "momoshop",
            "flag": "searchEngine",
            "data": {
                "searchValue": keyword,
                "searchType": "1",
                "cateLevel": "-1",
                "curPage": str(page),
                "priceS": "0",
                "priceE": "9999999",
                "showType": "chessboardType",
                "isFuzzy": "0",
                "rtnCateDatainfo": {
                    "cateCode": "",
                    "cateLv": "-1",
                    "keyword": keyword,
                    "curPage": str(page),
                    "historyDoPush": "false",
                    "timestamp": int(time.time() * 1000),
                },
                "flag": 2018,
                "serviceCode": "MT01",
                "addressSearchData": {},
            },
        }
        resp = session.post(MOMO_API, headers=MOMO_HEADERS, json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"第 {page} 頁 HTTP {resp.status_code}：{resp.text[:200]}")
        products = resp.json().get("rtnSearchData", {}).get("goodsInfoList", [])
        if not products:
            break
        for p in products:
            code = p.get("goodsCode", "")
            rows.append(
                {
                    "頁數": page,
                    "商品代碼": code,
                    "品名": p.get("goodsName", ""),
                    "價格": parse_price(p.get("goodsPrice", "")),
                    "連結": f"https://www.momoshop.com.tw/goods/GoodsDetail.jsp?i_code={code}"
                    if code
                    else "",
                }
            )
        report("MOMO", page, len(rows))
        if page < pages:
            time.sleep(delay)
    return finalize(rows, "MOMO")


def crawl_pchome(keyword: str, pages: int, delay: float, report) -> pd.DataFrame:
    session = requests.Session(impersonate="chrome")
    rows = []
    for page in range(1, pages + 1):
        resp = session.get(
            PCHOME_API,
            params={"q": keyword, "page": page, "sort": "sale/dc"},
            headers=PCHOME_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        prods = data.get("prods") or []
        if not prods:
            break
        for p in prods:
            pid = p.get("Id", "")
            rows.append(
                {
                    "頁數": page,
                    "商品代碼": pid,
                    "品名": p.get("name", ""),
                    "價格": p.get("price"),
                    "連結": f"https://24h.pchome.com.tw/prod/{pid}" if pid else "",
                }
            )
        report("PChome", page, len(rows))
        total = int(data.get("totalPage", 0) or 0)
        if total and page >= total:
            break
        if page < pages:
            time.sleep(delay)
    return finalize(rows, "PChome")


CRAWLERS = {"MOMO": crawl_momo, "PChome": crawl_pchome}


def run_crawl(keyword: str, pages: int, delay: float, platforms: list):
    """多平台平行爬取；UI 只在主執行緒更新。回傳 (結果dict, 錯誤dict)。"""
    cache = st.session_state.setdefault("_cache", {})
    now = time.time()
    results, errors, todo = {}, {}, []
    for p in platforms:
        hit = cache.get((p, keyword, pages))
        if hit and now - hit[0] < CACHE_TTL:
            results[p] = hit[1]
        else:
            todo.append(p)

    if todo:
        q: "queue.Queue" = queue.Queue()
        bar = st.progress(0.0, text="準備連線中…")
        done = {p: 0 for p in todo}
        total = pages * len(todo)

        def refresh():
            while True:
                try:
                    plat, page, n = q.get_nowait()
                except queue.Empty:
                    break
                done[plat] = page
                bar.progress(
                    min(sum(done.values()) / total, 1.0),
                    text=f"{plat} 已抓取第 {page}/{pages} 頁（累計 {n} 筆）",
                )

        with ThreadPoolExecutor(max_workers=len(todo)) as ex:
            futs = {
                p: ex.submit(CRAWLERS[p], keyword, pages, delay, lambda *a: q.put(a))
                for p in todo
            }
            while not all(f.done() for f in futs.values()):
                refresh()
                time.sleep(0.15)
        refresh()
        for p, f in futs.items():
            try:
                df = f.result()
                results[p] = df
                cache[(p, keyword, pages)] = (now, df)
            except Exception as e:  # 單一平台失敗不影響另一個
                errors[p] = str(e)
        bar.progress(1.0, text="抓取完成")
        time.sleep(0.3)
        bar.empty()
    return results, errors


# ------------------------------------------------------------------
# 資料處理
# ------------------------------------------------------------------
def apply_filters(df, price_range, include, exclude, drop_outliers):
    df = df[df["價格"].between(*price_range)]
    for w in include.split():
        df = df[df["品名"].str.contains(re.escape(w), case=False, na=False)]
    for w in exclude.split():
        df = df[~df["品名"].str.contains(re.escape(w), case=False, na=False)]
    if drop_outliers and not df.empty:
        q1 = df.groupby("平台")["價格"].transform(lambda s: s.quantile(0.25))
        q3 = df.groupby("平台")["價格"].transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        df = df[(df["價格"] >= q1 - 1.5 * iqr) & (df["價格"] <= q3 + 1.5 * iqr)]
    return df.reset_index(drop=True)


def add_price_range(df: pd.DataFrame, step: int):
    df = df.copy()
    lower = (df["價格"] // step) * step
    df["區間下限"] = lower
    df["價格區間"] = lower.map(lambda v: f"{v:,}~{v + step - 1:,}")
    order = (
        df.drop_duplicates("價格區間").sort_values("區間下限")["價格區間"].tolist()
    )
    return df, order


@st.cache_data(show_spinner=False)
def match_products(a: pd.DataFrame, b: pd.DataFrame, thr: float) -> pd.DataFrame:
    """以字元 bigram Jaccard 相似度配對 MOMO(a) 與 PChome(b) 的同款商品（一對一）。
    若兩邊品名都含 3 位數以上數字（容量/型號），則至少要有一個相同才算候選。"""

    def prep(df):
        out = []
        for name in df["品名"]:
            s = re.sub(r"[\W_]+", "", str(name).lower())
            out.append(({s[i : i + 2] for i in range(len(s) - 1)}, set(re.findall(r"\d{3,}", s))))
        return out

    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    pa, pb = prep(a), prep(b)
    cands = []
    for i, (ga, na) in enumerate(pa):
        if not ga:
            continue
        for j, (gb, nb) in enumerate(pb):
            if not gb or (na and nb and not (na & nb)):
                continue
            inter = len(ga & gb)
            if inter == 0:
                continue
            score = inter / (len(ga) + len(gb) - inter)
            if score >= thr:
                cands.append((score, i, j))
    cands.sort(reverse=True)

    used_a, used_b, pairs = set(), set(), []
    for score, i, j in cands:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((score, i, j))

    if not pairs:
        return pd.DataFrame()
    rows = []
    for score, i, j in pairs:
        ra, rb = a.iloc[i], b.iloc[j]
        diff = int(ra["價格"] - rb["價格"])
        rows.append(
            {
                "MOMO品名": ra["品名"],
                "MOMO價格": int(ra["價格"]),
                "PChome品名": rb["品名"],
                "PChome價格": int(rb["價格"]),
                "價差": diff,  # 正值 = PChome 較便宜
                "較便宜": "PChome" if diff > 0 else ("MOMO" if diff < 0 else "相同"),
                "相似度": round(score, 2),
                "MOMO連結": ra["連結"],
                "PChome連結": rb["連結"],
            }
        )
    return pd.DataFrame(rows).sort_values("價差", key=abs, ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------
# 圖表
# ------------------------------------------------------------------
def style(fig, height=600):
    fig.update_layout(
        font=dict(family=CJK_FONT, size=14),
        height=height,
        hovermode="closest",
        legend_title_text="平台",
        margin=dict(t=60, b=40),
    )
    return fig


def fig_distribution(df, log_x):
    fig = px.histogram(
        df, x="價格", color="平台", barmode="overlay", opacity=0.6, nbins=40,
        marginal="box", color_discrete_map=COLORS, hover_data=["品名"],
        title="價格分布（可拖曳縮放、點圖例切換平台）",
    )
    if log_x:
        fig.update_xaxes(type="log")
    return style(fig, 560)


def fig_range_bar(df, order):
    cnt = df.groupby(["價格區間", "平台"]).size().reset_index(name="商品數")
    fig = px.bar(
        cnt, x="價格區間", y="商品數", color="平台", barmode="group", text="商品數",
        category_orders={"價格區間": order}, color_discrete_map=COLORS,
        title="各價格區間商品數量",
    )
    fig.update_traces(textposition="outside")
    return style(fig, 460)


def fig_line(df, kw):
    fig = px.line(
        df, x="序號", y="價格", color="平台", markers=True, hover_name="品名",
        color_discrete_map=COLORS, labels={"序號": "搜尋結果順序", "價格": "價格 (NT$)"},
        title=f"「{kw}」售價折線圖（下方滑桿可縮放範圍）",
    )
    for plat, g in df.groupby("平台"):
        m = g["價格"].mean()
        fig.add_hline(
            y=m, line_dash="dash", line_color=COLORS.get(plat, "gray"),
            annotation_text=f"{plat} 平均 {m:,.0f}", annotation_position="top left",
        )
    fig.update_xaxes(rangeslider_visible=True)
    return style(fig, 650)


def fig_pie(sub, order, plat):
    cnt = sub["價格區間"].value_counts().reindex(order).fillna(0).reset_index()
    cnt.columns = ["價格區間", "商品數"]
    cnt = cnt[cnt["商品數"] > 0]
    fig = px.pie(cnt, names="價格區間", values="商品數", hole=0.35, title=f"{plat} 價格區間占比")
    fig.update_traces(sort=False, textinfo="percent+label", pull=[0.03] * len(cnt))
    fig.update_layout(showlegend=False)
    return style(fig, 520)


def fig_sunburst(df, kw, top_n):
    d = df.groupby("平台", group_keys=False).head(top_n).copy()
    d["商品"] = d["品名"].str.slice(0, 18)
    d["商品"] += d.groupby(["平台", "價格區間", "商品"]).cumcount().map(
        lambda n: "" if n == 0 else f" ({n + 1})"
    )
    fig = px.sunburst(
        d, path=["平台", "價格區間", "商品"], color="平台", color_discrete_map=COLORS,
        hover_data={"品名": True, "價格": True},
        title=f"「{kw}」平台 → 價格區間 → 商品（點擊扇形可下鑽，每平台前 {top_n} 筆）",
    )
    return style(fig, 780)


def fig_match_scatter(m):
    colors = m["較便宜"].map({"MOMO": COLORS["MOMO"], "PChome": COLORS["PChome"]}).fillna("#999999")
    hi = max(m["MOMO價格"].max(), m["PChome價格"].max()) * 1.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", name="兩平台等價",
                             line=dict(dash="dash", color="gray"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=m["MOMO價格"], y=m["PChome價格"], mode="markers", name="配對商品",
        marker=dict(size=10, color=colors, line=dict(width=1, color="white"), opacity=0.85),
        customdata=np.stack([m["MOMO品名"], m["PChome品名"], m["價差"], m["相似度"]], axis=-1),
        hovertemplate=(
            "<b>MOMO</b>：%{customdata[0]}<br><b>PChome</b>：%{customdata[1]}<br>"
            "MOMO $%{x:,}　PChome $%{y:,}<br>價差(MOMO−PChome)：%{customdata[2]:,}<br>"
            "相似度：%{customdata[3]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="同款商品比價散佈圖（對角線上方 = MOMO 較便宜；可框選/套索選取）",
        xaxis_title="MOMO 價格 (NT$)", yaxis_title="PChome 價格 (NT$)", dragmode="select",
        showlegend=False,
    )
    return style(fig, 620)


# ------------------------------------------------------------------
# 介面：側邊欄查詢條件
# ------------------------------------------------------------------
st.title("🛒 MOMO × PChome 比價系統")
st.caption("輸入關鍵字與頁數，同時抓取兩大平台並比較價格；資料僅供參考，實際售價以各平台為準。")

with st.sidebar:
    st.header("🔎 查詢條件")
    keyword = st.text_input("商品關鍵字", value="行動電源")
    pages = st.number_input("爬取頁數（每平台）", min_value=1, max_value=20, value=2, step=1)
    platforms = st.multiselect("比價平台", list(CRAWLERS), default=list(CRAWLERS))
    delay = st.slider("每頁間隔秒數", 0.5, 5.0, 1.0, 0.5, help="間隔越長越不容易被網站封鎖")
    fx = st.toggle("🎉 完成時顯示特效", value=True)
    run = st.button("🚀 開始比價", type="primary", **STRETCH)

if run:
    if not keyword.strip():
        st.warning("請輸入商品關鍵字")
    elif not platforms:
        st.warning("請至少選擇一個平台")
    else:
        results, errors = run_crawl(keyword.strip(), int(pages), delay, platforms)
        for p, msg in errors.items():
            st.error(f"{p} 爬取失敗：{msg}")
        results = {p: d for p, d in results.items() if not d.empty}
        if results:
            st.session_state["data"] = results
            st.session_state["keyword"] = keyword.strip()
            summary = "、".join(f"{p} {len(d)} 筆" for p, d in results.items())
            st.toast(f"完成！{summary}", icon="✅")
            if fx:
                st.balloons()
        else:
            st.session_state.pop("data", None)
            st.warning("查無資料，請換個關鍵字或稍後再試。")

data = st.session_state.get("data")
kw = st.session_state.get("keyword", "")
if not data:
    st.info("請在左側輸入關鍵字、頁數並選擇平台，然後按下「開始比價」。")
    st.stop()

raw = pd.concat(data.values(), ignore_index=True)

# ------------------------------------------------------------------
# 側邊欄：即時篩選（不需重新爬取）
# ------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.header("🎛️ 即時篩選")
    lo, hi = int(raw["價格"].min()), int(raw["價格"].max())
    price_range = st.slider("價格範圍 (NT$)", lo, hi, (lo, hi)) if lo < hi else (lo, hi)
    include = st.text_input("品名須包含（空白分隔，皆須符合）", placeholder="例如：10000 PD")
    exclude = st.text_input("品名排除字詞", placeholder="例如：保護殼 線材")
    drop_outliers = st.checkbox("排除極端值（IQR）", value=False)
    step = st.select_slider("價格區間級距 (NT$)", [100, 200, 300, 500, 1000, 2000, 5000], value=500)
    top_n = st.slider("旭日圖每平台商品數", 20, 300, 60, 10)

df = apply_filters(raw, price_range, include, exclude, drop_outliers)
if df.empty:
    st.warning("篩選後沒有商品，請放寬條件。")
    st.stop()
df, range_order = add_price_range(df, step)
plats = [p for p in CRAWLERS if p in df["平台"].unique()]

# ------------------------------------------------------------------
# 各平台摘要卡片 + 勝負判定
# ------------------------------------------------------------------
cols = st.columns(len(plats))
stats = {}
for col, p in zip(cols, plats):
    s = df.loc[df["平台"] == p, "價格"]
    stats[p] = dict(n=len(s), mean=s.mean(), median=s.median(), min=s.min(), max=s.max())
    with col.container(border=True):
        st.markdown(f"<h4 style='color:{COLORS[p]};margin:0'>● {p}</h4>", unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("商品數", f"{stats[p]['n']:,}")
        b.metric("平均價", f"{stats[p]['mean']:,.0f}")
        c.metric("中位數", f"{stats[p]['median']:,.0f}")
        d, e = st.columns(2)
        d.metric("最低價", f"{stats[p]['min']:,}")
        e.metric("最高價", f"{stats[p]['max']:,}")

if len(plats) == 2:
    a, b = plats
    lines = []
    for label, key in (("平均價", "mean"), ("中位數", "median")):
        va, vb = stats[a][key], stats[b][key]
        win, lose = (a, b) if va < vb else (b, a)
        pct = abs(va - vb) / max(va, vb) * 100
        lines.append(f"**{label}**：{win} 較低（便宜約 {pct:.1f}%）")
    st.success("　｜　".join(lines))

# ------------------------------------------------------------------
# 分頁
# ------------------------------------------------------------------
tabs = st.tabs(["📊 比價總覽", "📈 折線圖", "🥧 圓餅圖", "🌞 旭日圖", "🔗 同款比對", "🏆 最低價榜", "📋 明細下載"])

with tabs[0]:
    log_x = st.toggle("價格軸使用對數刻度", value=False)
    st.plotly_chart(fig_distribution(df, log_x), config=PLOT_CFG, **STRETCH)
    st.plotly_chart(fig_range_bar(df, range_order), config=PLOT_CFG, **STRETCH)

with tabs[1]:
    st.plotly_chart(fig_line(df, kw), config=PLOT_CFG, **STRETCH)

with tabs[2]:
    pcols = st.columns(len(plats))
    for col, p in zip(pcols, plats):
        col.plotly_chart(fig_pie(df[df["平台"] == p], range_order, p), config=PLOT_CFG, **STRETCH)

with tabs[3]:
    st.plotly_chart(fig_sunburst(df, kw, top_n), config=PLOT_CFG, **STRETCH)

with tabs[4]:
    if len(plats) < 2:
        st.info("同款比對需要同時選擇 MOMO 與 PChome 兩個平台。")
    else:
        thr = st.slider("相似度門檻（越高越嚴格）", 0.30, 0.90, 0.50, 0.05)
        m = match_products(
            df[df["平台"] == "MOMO"][["品名", "價格", "連結"]],
            df[df["平台"] == "PChome"][["品名", "價格", "連結"]],
            thr,
        )
        if m.empty:
            st.warning("找不到符合門檻的同款商品，請調低相似度或增加爬取頁數。")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("配對成功", f"{len(m):,} 組")
            c2.metric("MOMO 較便宜", f"{(m['較便宜'] == 'MOMO').sum()} 組")
            c3.metric("PChome 較便宜", f"{(m['較便宜'] == 'PChome').sum()} 組")
            c4.metric("平均價差", f"{m['價差'].abs().mean():,.0f}")

            event = st.plotly_chart(
                fig_match_scatter(m), config=PLOT_CFG, key="match_scatter",
                on_select="rerun", selection_mode=("points", "box", "lasso"), **STRETCH,
            )
            try:
                idx = [pt["point_index"] for pt in event["selection"]["points"] if pt.get("curve_number", 1) == 1]
            except Exception:
                idx = []
            view = m.iloc[idx] if idx else m
            st.subheader(f"配對明細（{'已選取 ' + str(len(idx)) + ' 筆' if idx else '全部'}）")
            st.dataframe(
                view, hide_index=True, **STRETCH,
                column_config={
                    "MOMO連結": st.column_config.LinkColumn("MOMO", display_text="前往"),
                    "PChome連結": st.column_config.LinkColumn("PChome", display_text="前往"),
                    "相似度": st.column_config.ProgressColumn("相似度", min_value=0, max_value=1, format="%.2f"),
                    "價差": st.column_config.NumberColumn("價差(MOMO−PChome)", format="%d"),
                },
            )

with tabs[5]:
    n = st.slider("顯示最低價前幾名", 5, 50, 15)
    low = df.nsmallest(n, "價格").copy()
    low["簡稱"] = low["品名"].str.slice(0, 24) + " [" + low["平台"] + "]"
    fig = px.bar(
        low.iloc[::-1], x="價格", y="簡稱", color="平台", orientation="h", text="價格",
        color_discrete_map=COLORS, hover_name="品名", title=f"「{kw}」最低價前 {n} 名",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(style(fig, max(420, n * 28)), config=PLOT_CFG, **STRETCH)

with tabs[6]:
    q = st.text_input("🔍 在結果中搜尋品名", key="detail_q")
    view = df[df["品名"].str.contains(re.escape(q), case=False, na=False)] if q else df
    show = view[["平台", "頁數", "品名", "價格", "價格區間", "連結"]]
    st.dataframe(
        show, hide_index=True, **STRETCH,
        column_config={
            "連結": st.column_config.LinkColumn("連結", display_text="前往"),
            "價格": st.column_config.NumberColumn("價格", format="NT$ %d"),
        },
    )
    st.download_button(
        "⬇️ 下載 CSV",
        show.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"比價_{kw}_{time.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
