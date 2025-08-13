def compute_diff(dfA: pd.DataFrame, dfB: pd.DataFrame, keys: list[str], compare_cols: list[str]):
    # 正規化
    A = normalize(dfA[keys + compare_cols])
    B = normalize(dfB[keys + compare_cols])

    # 以鍵建索引
    A = A.drop_duplicates(subset=keys).set_index(keys, drop=False)
    B = B.drop_duplicates(subset=keys).set_index(keys, drop=False)

    idxA, idxB = set(A.index.tolist()), set(B.index.tolist())
    added_keys   = sorted(list(idxB - idxA))
    removed_keys = sorted(list(idxA - idxB))
    common_keys  = sorted(list(idxA & idxB))

    # 新增/刪除
    added   = B.loc[added_keys] if added_keys else pd.DataFrame(columns=A.columns)
    removed = A.loc[removed_keys] if removed_keys else pd.DataFrame(columns=A.columns)

    changed_wide = pd.DataFrame()
    changed_long = pd.DataFrame(columns=keys + ["column","value_A","value_B"])

    if common_keys and compare_cols:
        A_common = A.loc[common_keys, compare_cols]
        B_common = B.loc[common_keys, compare_cols]
        neq = (A_common != B_common)
        idx_changed = neq.index[neq.any(axis=1)]  # ✅ 用索引定位變更列

        if len(idx_changed) > 0:
            # 寬表
            keys_df = A.loc[idx_changed, keys].reset_index(drop=True)
            wideA   = A_common.loc[idx_changed].add_suffix("_A").reset_index(drop=True)
            wideB   = B_common.loc[idx_changed].add_suffix("_B").reset_index(drop=True)
            changed_wide = pd.concat([keys_df, wideA, wideB], axis=1)

            # 標示變更欄位
            changed_wide["changed_cols"] = [
                ", ".join(row.index[row].tolist()) for _, row in neq.loc[idx_changed].iterrows()
            ]

            # 長表
            recs = []
            for key_vals, row in neq.loc[idx_changed].iterrows():
                for c in row.index[row.values]:
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
                changed_long = pd.DataFrame.from_records(recs, columns=keys+["column","value_A","value_B"])

    return added.reset_index(drop=True), removed.reset_index(drop=True), changed_wide, changed_long
