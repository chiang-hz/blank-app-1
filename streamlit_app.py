# -*- coding: utf-8 -*-
from __future__ import annotations

import os, sqlite3, tempfile, re
import pandas as pd
import streamlit as st

# --------------------------
# 基本設定（可在 UI 調整）
# --------------------------
IGNORE_REGEX_DEFAULT = r"^(updated_at|created_at|last_modified|modify_time|timestamp|ts|version)$"
CASE_INSENSITIVE_DEFAULT = True   # 比對前轉小寫
TRIM_WHITESPACE_DEFAULT  = True   # 比對前去除前後空白

st.set_page_config(page_title="兩庫全表全欄比對", layout="wide")
st.title("兩個 SQLite 資料庫：全表 × 全欄 差異比對")

# 這三個變數會在 UI 裡被更新（不需要 global）
IGNORE_REGEX = IGNORE_REGEX_DEFAULT
CASE_INSENSITIVE = CASE_INSENSITIVE_DEFAULT
TRIM_WHITESPACE = TRIM_WHITESPACE_DEFAULT


# --------------------------
# 工具函式
# --------------------------
def qident(name: str) -> str:
    """SQLite 識別字引用（支援中文 / 空白 / 特殊字元）"""
    return '"' + str(name).replace('"', '""') + '"'

def connect_db(path_or_file) -> sqlite3.Connection:
    return sqlite3.connect(path_or_file)

@st.cache_data(show_spinner=False)
def list_tables(db_path: str) -> list[str]:
    conn = connect_db(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY 1", conn
        )
        return df["name"].tolist()
    finally:
        conn.close()

@st.cache_data(show_spinner=False)
def table_info(db_path: str, table: str) -> pd.DataFrame:
    """PRAGMA table_info：取得欄位名稱、型別、是否主鍵等"""
    conn = connect_db(db_path)
    try:
        return pd.read_sql_query(f"PRAGMA table_info({qident(table)})", conn)
    finally:
        conn.close()

def unique_indexes(db_path: str, table: str) -> list[list[str]]:
    """取得唯一索引（可能為複合欄），回傳每個索引的欄位清單"""
    conn = connect_db(db_path)
    try:
        idx_df = pd.read_sql_query(f"PRAGMA index_list({qident(table)})", conn)
        uniques = []
        if not idx_df.empty:
            # 某些版本欄名為 'unique' 或 'origin'，保守處理
            candidate_names = []
            for _, r in idx_df.iterrows():
                is_unique = False
                if "unique" in r and int(r["unique"]) == 1:
                    is_unique = True
                elif "origin" in r and str(r["origin"]).lower() == "u":
                    is_unique = True
                if is_unique and "name" in r:
                    candidate_names.append(r["name"])
            for idx_name in candidate_names:
                info = pd.read_sql_query(f"PRAGMA index_info({qident(idx_name)})", conn)
                cols = info.sort_values("seqno")["name"].tolist() if "name" in info else []
                if cols:
                    uniques.append(cols)
        return uniques
    finally:
        conn.close()

def choose_key(dbA: str, dbB: str, table: str) -> list[str]:
    """自動選擇鍵：主鍵 > 共同唯一索引 > 常見鍵名 > 無鍵"""
    infoA, infoB = table_info(dbA, table), table_info(dbB, table)
    # 1) 主鍵（欄位集合相同即採用）
    pkA = infoA[infoA["pk"] > 0].sort_values("pk")["name"].tolist()
    pkB = infoB[infoB["pk"] > 0].sort_values("pk")["name"].tolist()
    if pkA and pkB and set(pkA) == set(pkB):
        return pkA
    # 2) 共同唯一索引
    uniqA = [set(u) for u in unique_indexes(dbA, table)]
    uniqB = [set(u) for u in unique_indexes(dbB, table)]
    for ua in uniqA:
        if ua in uniqB:
            return list(ua)
    # 3) Heuristic：常見鍵名（尾碼 id/code/編號/代碼）
    colsA, colsB = set(infoA["name"]), set(infoB["name"])
    common = sorted(list(colsA & colsB))
    pat = re.compile(r"(id|code|編號|代碼)$", re.IGNORECASE)
    guess = [c for c in common if pat.search(str(c))]
    if guess:
        return guess[:2]
    # 4) 無鍵
    return []

def effective_common_cols(infoA: pd.DataFrame, infoB: pd.DataFrame, ignore_regex: str | None) -> list[str]:
    """求共同欄位並依 regex 忽略噪音欄位"""
    colsA = set(infoA["name"].tolist()); colsB = set(infoB["name"].tolist())
    common = sorted(list(colsA & colsB))
    if ignore_regex and ignore_regex.strip():
        # 忽略大小寫以提升實務容錯
        reg = re.compile(ignore_regex, re.IGNORECASE)
        common = [c for c in common if not reg.match(str(c))]
    return common

def fetch_all(db_path: str, table: str, include_cols: list[str]) -> pd.DataFrame:
    conn = connect_db(db_path)
    try:
        sel = ", ".join(qident(c) for c in include_cols)
        return pd.read_sql_query(f"SELECT {sel} FROM {qident(table)}", conn)
    finally:
        conn.close()

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """一致化資料：轉字串、trim、大小寫、NaN/None → 空字串"""
    out = df.copy()
    for c in out.columns:
        # 先轉為字串
        out[c] = out[c].astype(str)
        if TRIM_WHITESPACE:
            out[c] = out[c].str.strip()
        # 視 None/nan 為空
        out[c] = out[c].replace({"None": "", "none": "", "NaN": "", "nan": ""})
        if CASE_INSENSITIVE:
            out[c] = out[c].str.lower()
    return out

def compute_diff(dfA: pd.DataFrame, dfB: pd.DataFrame, keys: list[str], compare_cols: list[str]):
    """
    回傳：added, removed, changed_wide, changed_long
    - added/removed：依鍵比對
    - changed_wide：鍵 + 各欄位_A/_B + changed_cols
    - changed_long：鍵 + column + value_A + value_B（逐欄明細）
    """
    # 正規化並依鍵建索引
    A = normalize(dfA[keys + compare_cols])
    B = normalize(dfB[keys + compare_cols])
    A = A.drop_duplicates(subset=keys).set_index(keys, drop=False)
    B = B.drop_duplicates(subset=keys).set_index(keys, drop=False)

    idxA, idxB = set(A.index.tolist()), set(B.index.tolist())
    added_keys   = sorted(list(idxB - idxA))
    removed_keys = sorted(list(idxA - idxB))
    common_keys  = sorted(list(idxA & idxB))

    # 新增 / 刪除
    added   = B.loc[added_keys] if added_keys else pd.DataFrame(columns=A.columns)
    removed = A.loc[removed_keys] if removed_keys else pd.DataFrame(columns=A.columns)

    changed_wide = pd.DataFrame()
    changed_long = pd.DataFrame(columns=keys + ["column", "value_A", "value_B"])

    if common_keys and compare_cols:
        # 只取共同鍵的比對欄位
        A_common = A.loc[common_keys, compare_cols]
        B_common = B.loc[common_keys, compare_cols]

        neq = (A_common != B_common)
        idx_changed = neq.index[neq.any(axis=1)]  # 用索引定位（避免 Unalignable boolean Series）

        if len(idx_changed) > 0:
            # 寬表：鍵 + 每欄的 _A/_B
            keys_df = A.loc[idx_changed, keys].reset_index(drop=True)
            wideA   = A_common.loc[idx_changed].add_suffix("_A").reset_index(drop=True)
            wideB   = B_common.loc[idx_changed].add_suffix("_B").reset_index(drop=True)
            changed_wide = pd.concat([keys_df, wideA, wideB], axis=1)

            # 標記有哪些欄位變更
            changed_wide["changed_cols"] = [
                ", ".join(row.index[row].tolist()) for _, row in neq.loc[idx_changed].iterrows()
            ]

            # 長表：逐欄差異
            recs = []
            for key_vals, row in neq.loc[idx_changed].iterrows():
                diff_cols = row.index[row.values].tolist()
                for c in diff_cols:
                    rec = {}
                    if isinstance(key_vals, tuple):
                        for kname, kval in zip(keys, key_vals): rec[kname] = kval
                    else:
                        rec[keys[0]] = key_vals
                    rec["column"]  = c
                    rec["value_A"] = A.loc[key_vals, c]
                    rec["value_B"] = B.loc[key_vals, c]
                    recs.append(rec)
            if recs:
                changed_long = pd.DataFrame.from_records(recs, columns=keys + ["column", "value_A", "value_B"])

    return added.reset_index(drop=True), removed.reset_index(drop=True), changed_wide, changed_long

def rowset_diff_no_key(dfA: pd.DataFrame, dfB: pd.DataFrame):
    """無鍵可用時：以整列字串簽章做集合差（只能新增/刪除，無法判斷變更）"""
    A = normalize(dfA)
    B = normalize(dfB)
    A["_sig"] = A.apply(lambda r: "||".join(r.values.tolist()), axis=1)
    B["_sig"] = B.apply(lambda r: "||".join(r.values.tolist()), axis=1)
    setA, setB = set(A["_sig"]), set(B["_sig"])
    added   = B[B["_sig"].isin(setB - setA)].drop(columns=["_sig"])
    removed = A[A["_sig"].isin(setA - setB)].drop(columns=["_sig"])
    return added.reset_index(drop=True), removed.reset_index(drop=True)


# --------------------------
# UI：選兩個資料庫（上傳或路徑）
# --------------------------
st.caption("可上傳 .db/.sqlite，或直接填路徑")
l, r = st.columns(2)
with l:
    upA  = st.file_uploader("資料庫 A", type=["db","sqlite"], key="dbA")
    pathA = st.text_input("或輸入 A 路徑", value="cbc_financial_data (113審).db")
with r:
    upB  = st.file_uploader("資料庫 B", type=["db","sqlite"], key="dbB")
    pathB = st.text_input("或輸入 B 路徑", value="cbc_financial_data (113審).db")

# 轉存上傳檔為暫存檔
tmpA = tmpB = None
if upA:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    f.write(upA.read()); f.flush(); tmpA = f.name; f.close()
if upB:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    f.write(upB.read()); f.flush(); tmpB = f.name; f.close()

dbA = tmpA or pathA
dbB = tmpB or pathB

if not (upA or os.path.exists(dbA)):
    st.error("找不到 A 資料庫檔案。")
    st.stop()
if not (upB or os.path.exists(dbB)):
    st.error("找不到 B 資料庫檔案。")
    st.stop()

# 共同資料表
tablesA, tablesB = set(list_tables(dbA)), set(list_tables(dbB))
common_tables = sorted(list(tablesA & tablesB))
if not common_tables:
    st.error("兩個資料庫沒有共同的資料表。")
    st.stop()

st.success(f"共找到 {len(common_tables)} 個共同資料表。")

# 進階選項：忽略欄位、大小寫、空白處理
with st.expander("進階選項（可略過）", expanded=False):
    ignore_regex = st.text_input("忽略欄位（正則）", value=IGNORE_REGEX_DEFAULT,
                                 help="符合此正則的欄位將不參與比對，例如時間戳、版本號等噪音欄位。")
    case_insensitive = st.checkbox("比對忽略大小寫", value=CASE_INSENSITIVE_DEFAULT)
    trim_whitespace  = st.checkbox("比對前去除前後空白", value=TRIM_WHITESPACE_DEFAULT)

# 更新全域設定（不需 global）
IGNORE_REGEX = ignore_regex
CASE_INSENSITIVE = case_insensitive
TRIM_WHITESPACE = trim_whitespace

# 執行比對
run = st.button("開始比對所有共同表")
if run:
    summary_rows = []
    per_table_results: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]] = {}

    for t in common_tables:
        infoA = table_info(dbA, t)
        infoB = table_info(dbB, t)
        common_cols = effective_common_cols(infoA, infoB, IGNORE_REGEX)

        if not common_cols:
            summary_rows.append({"table": t, "added": 0, "removed": 0, "changed": 0, "note": "無共同欄位"})
            per_table_results[t] = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [])
            continue

        # 取資料
        dfA = fetch_all(dbA, t, common_cols)
        dfB = fetch_all(dbB, t, common_cols)

        # 自動決定鍵
        keys = choose_key(dbA, dbB, t)
        # 僅保留共同欄位內存在的鍵
        keys = [k for k in keys if k in common_cols]
        compare_cols = [c for c in common_cols if c not in set(keys)]

        # 比對
        if keys:
            added, removed, changed_wide, changed_long = compute_diff(dfA, dfB, keys, compare_cols)
            summary_rows.append({
                "table": t,
                "added": len(added),
                "removed": len(removed),
                "changed": len(changed_wide),
                "note": f"鍵: {', '.join(keys)}；比對欄位數: {len(compare_cols)}"
            })
            per_table_results[t] = (added, removed, changed_wide, changed_long, keys, compare_cols)
        else:
            added, removed = rowset_diff_no_key(dfA[common_cols], dfB[common_cols])
            summary_rows.append({
                "table": t,
                "added": len(added),
                "removed": len(removed),
                "changed": 0,
                "note": "無可用鍵；僅做新增/刪除"
            })
            per_table_results[t] = (added, removed, pd.DataFrame(), pd.DataFrame(), [], common_cols)

    # 總覽
    st.subheader("比對摘要")
    summary = pd.DataFrame(summary_rows).sort_values(["added", "removed", "changed"], ascending=False)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # 檢視各表詳情
    st.subheader("選擇資料表查看詳情")
    pick = st.selectbox("資料表", options=[r["table"] for r in summary_rows])

    added, removed, changed_wide, changed_long, keys, compare_cols = per_table_results[pick]
    m1, m2, m3 = st.columns(3)
    m1.metric("新增（B 有 A 無）", len(added))
    m2.metric("刪除（A 有 B 無）", len(removed))
    m3.metric("變更（鍵相同欄位不同）", len(changed_wide))

    tabs = st.tabs(["新增", "刪除", "變更（寬表）", "變更（長表）"])
    with tabs[0]:
        if added.empty: st.info("無新增")
        else:
            st.dataframe(added, use_container_width=True)
            st.download_button("下載新增 CSV", data=added.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{pick}_added.csv", mime="text/csv")
    with tabs[1]:
        if removed.empty: st.info("無刪除")
        else:
            st.dataframe(removed, use_container_width=True)
            st.download_button("下載刪除 CSV", data=removed.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{pick}_removed.csv", mime="text/csv")
    with tabs[2]:
        if changed_wide.empty: st.info("無變更或無鍵")
        else:
            st.dataframe(changed_wide, use_container_width=True)
            st.download_button("下載變更（寬表）CSV", data=changed_wide.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{pick}_changed_wide.csv", mime="text/csv")
    with tabs[3]:
        if changed_long.empty: st.info("無變更或無鍵")
        else:
            st.dataframe(changed_long, use_container_width=True)
            st.download_button("下載變更（長表）CSV", data=changed_long.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{pick}_changed_long.csv", mime="text/csv")

# 收尾：刪除暫存檔
for p in [tmpA, tmpB]:
    if p and os.path.exists(p):
        try:
            os.remove(p)
        except Exception:
            pass
