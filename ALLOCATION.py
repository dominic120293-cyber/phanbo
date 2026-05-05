"""
Phiên bản tối ưu tốc độ – v4.2 (CBC Stable)
Đã bỏ HiGHS để tránh lỗi trên server
"""

import io
import time
import os
import pandas as pd
import pulp
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from collections import defaultdict

# ============================================================
# TUNING FLAGS
# ============================================================
ULTRA_FAST = True
SOLVER_TIME_LIMIT = 120
MIP_GAP = 0.05

# ============================================================
# COLOR & STYLE
# ============================================================
C_DARK_BLUE   = "FF1F4E79"
C_MID_BLUE    = "FF2E75B6"
C_LIGHT_BLUE  = "FF9DC3E6"
C_PALE_BLUE   = "FFD6E4F0"
C_ALT_ROW     = "FFEBF3FB"
C_WHITE       = "FFFFFFFF"

FONT_NAME = "Calibri"

def _font(bold=False, color="FF000000", size=10):
    return Font(name=FONT_NAME, bold=bold, color=color, size=size)

def _fill(color):
    return PatternFill("solid", fgColor=color)

def _align(h="center", v="center", wrap=False):
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)

def _thin_border():
    s = Side(border_style="thin", color="FF000000")
    return Border(left=s, right=s, top=s, bottom=s)

# ============================================================
# SOLVER - CHỈ DÙNG CBC (đã fix lỗi highs)
# ============================================================
def _n_threads():
    try:
        return max(1, os.cpu_count() or 1)
    except:
        return 1

def _make_solver(time_limit=120):
    n = _n_threads()
    solver = pulp.PULP_CBC_CMD(
        msg=False,           # Tắt log để nhanh
        timeLimit=time_limit,
        threads=n,
        options=['ratioGap 0.05']   # Cho phép gap 5% để tăng tốc
    )
    print(f"[Solver] CBC ({n} threads, gap=5%) - Stable")
    return solver

# ============================================================
# HOUR SORT KEY
# ============================================================
_DAY_RANK = {'MO': 0, 'TU': 1, 'WE': 2, 'TH': 3, 'FR': 4, 'SA': 5, 'SU': 6}

def _hour_sort_key(h: str):
    s = str(h).strip()
    if s.startswith('+'):
        s = s[1:]
    if len(s) >= 2:
        day_code = s[:2].upper()
        time_str = s[2:].strip()
        day_rank = _DAY_RANK.get(day_code, 99)
        try:
            time_int = int(time_str) if time_str else 0
        except ValueError:
            time_int = 0
        return (day_rank, time_int)
    return (99, s)

# ============================================================
# MAIN FUNCTION
# ============================================================
def run_optimization(file_input):
    t0 = time.perf_counter()

    # ============================================================
    # 1. ĐỌC DỮ LIỆU
    # ============================================================
    xls = pd.ExcelFile(file_input)

    df1 = pd.read_excel(xls, sheet_name='MOVEHOUR-WEIGHTCLASS', header=None)
    has_st_pod = (str(df1.iloc[1, 2]).strip().upper() == 'ST')
    data_col_start = 4 if has_st_pod else 2

    sts_bay_map = {}
    for col in range(data_col_start, df1.shape[1]):
        sts = df1.iloc[0, col]
        bay = df1.iloc[1, col]
        if pd.notna(sts) and pd.notna(bay):
            sts_bay_map[col] = (str(sts).strip(), str(bay).strip())

    demands = {}
    current_hour = None
    for idx in range(2, df1.shape[0]):
        row = df1.iloc[idx]
        hour = row[0]
        if pd.isna(hour):
            hour = current_hour
        else:
            current_hour = hour
        weight = row[1]
        if pd.isna(weight):
            continue
        weight = int(float(str(weight)))
        st_val  = str(row[2]).strip() if has_st_pod and pd.notna(row[2]) else ''
        pod_val = str(row[3]).strip() if has_st_pod and pd.notna(row[3]) else ''
        for col in range(data_col_start, df1.shape[1]):
            qty = row[col]
            if pd.notna(qty) and qty != '':
                qty = int(float(str(qty)))
                if qty > 0:
                    sts, bay = sts_bay_map[col]
                    key = (hour, sts, bay)
                    demands.setdefault(key, {})
                    dkey = (weight, st_val, pod_val)
                    demands[key][dkey] = demands[key].get(dkey, 0) + qty

    job_keys = list(demands.keys())
    all_hours_sorted = sorted(set(h for (h, s, b) in job_keys), key=_hour_sort_key)
    hour_rank = {h: i for i, h in enumerate(all_hours_sorted)}

    # Supply
    df2 = pd.read_excel(xls, sheet_name='BLOCK-WEIGHT CLASS', header=0)
    col_names = [str(c).strip() for c in df2.columns]
    has_st_pod_supply = len(col_names) > 2 and col_names[1].upper() == 'ST'
    wc_col_start = 3 if has_st_pod_supply else 1

    supply = {}; blocks_set = set()
    for _, row in df2.iterrows():
        block = str(row.iloc[0]).strip()
        if block in ('nan', 'GRAND TOTAL', '') or not block: 
            continue
        st_v  = str(row.iloc[1]).strip() if has_st_pod_supply else ''
        pod_v = str(row.iloc[2]).strip() if has_st_pod_supply else ''
        skey = (block, st_v, pod_v)
        wc_dict = {}
        for wi, w in enumerate([1, 2, 3, 4, 5]):
            ci = wc_col_start + wi
            val = row.iloc[ci] if ci < len(row) else None
            wc_dict[w] = int(val) if pd.notna(val) and val != '' else 0
        supply[skey] = wc_dict
        blocks_set.add(block)

    blocks = sorted(blocks_set)
    supply_keys = [k for k in supply if any(supply[k][w] > 0 for w in [1,2,3,4,5])]

    supply_by_st_pod_wc = defaultdict(lambda: defaultdict(dict))
    for b, st_v, pod_v in supply_keys:
        for w in [1,2,3,4,5]:
            if supply[(b, st_v, pod_v)][w] > 0:
                supply_by_st_pod_wc[(st_v, pod_v, w)][b] = supply[(b, st_v, pod_v)][w]

    print(f"ULTRA_FAST={ULTRA_FAST} | Jobs={len(job_keys)} | Blocks={len(blocks)}")

    t_read = time.perf_counter()
    print(f"[TIMER] Đọc dữ liệu: {t_read - t0:.1f}s")

    # ============================================================
    # 2. MIP MODEL
    # ============================================================
    prob = pulp.LpProblem("Min_Clashes", pulp.LpMinimize)

    # y_vars tối ưu
    possible_blocks = defaultdict(set)
    for (h, s, bay), ddict in demands.items():
        for (w, st, pod) in ddict:
            for b in supply_by_st_pod_wc[(st, pod, w)]:
                possible_blocks[(h, s, bay)].add(b)

    y_vars = {(h, s, bay, b): pulp.LpVariable(f"y_{h}_{s}_{bay}_{b}", cat='Binary')
              for (h, s, bay) in job_keys for b in possible_blocks[(h, s, bay)]}

    x_vars = {}
    for (h, s, bay), ddict in demands.items():
        for dkey, d in ddict.items():
            w, st_v, pod_v = dkey
            for b in supply_by_st_pod_wc[(st_v, pod_v, w)]:
                key = (h, s, bay, b, dkey)
                x_vars[key] = pulp.LpVariable(f"x_{h}_{s}_{bay}_{b}_{w}", lowBound=0, cat='Integer')

    # u / e
    jobs_by_hour = defaultdict(list)
    for (h, s, bay) in job_keys:
        jobs_by_hour[h].append((s, bay))

    u_vars = {}; e_vars = {}
    for h in jobs_by_hour:
        for b in blocks:
            y_list = [y_vars[(h, s, bay, b)] for (s, bay) in jobs_by_hour[h] if (h, s, bay, b) in y_vars]
            if not y_list: continue
            u_vars[(h, b)] = pulp.LpVariable(f"u_{h}_{b}", lowBound=0, cat='Integer')
            e_vars[(h, b)] = pulp.LpVariable(f"e_{h}_{b}", lowBound=0, cat='Integer')
            prob += u_vars[(h, b)] == pulp.lpSum(y_list)
            prob += e_vars[(h, b)] >= u_vars[(h, b)] - 1

    # single_block
    single_block = {}
    for (h, s, bay) in job_keys:
        single_block[(h, s, bay)] = pulp.LpVariable(f"sb_{h}_{s}_{bay}", lowBound=0, upBound=1, cat='Continuous')
        ysum = pulp.lpSum(y_vars.get((h, s, bay, b), 0) for b in possible_blocks[(h, s, bay)])
        prob += single_block[(h, s, bay)] >= (2 - ysum)

    # Objective
    prob += (100 * pulp.lpSum(e_vars.values()) + 10 * pulp.lpSum(single_block.values()))

    # Demand
    for (h, s, bay), ddict in demands.items():
        for dkey, d in ddict.items():
            xlist = [x_vars[k] for k in x_vars if k[0]==h and k[1]==s and k[2]==bay and k[4]==dkey]
            if xlist:
                prob += pulp.lpSum(xlist) == d

    # Supply & Linking
    x_by_supply = defaultdict(list)
    for key, xvar in x_vars.items():
        b = key[3]
        w, st, pod = key[4]
        x_by_supply[(b, st, pod, w)].append(xvar)

    for (b, st_v, pod_v), wc_dict in supply.items():
        for w in [1,2,3,4,5]:
            xl = x_by_supply.get((b, st_v, pod_v, w), [])
            if xl:
                prob += pulp.lpSum(xl) <= wc_dict[w]

    for key, xvar in x_vars.items():
        h, s, bay, b, dkey = key
        d = demands[(h, s, bay)][dkey]
        prob += xvar <= d * y_vars[(h, s, bay, b)]

    t_build = time.perf_counter()
    print(f"[TIMER] Build model: {t_build - t_read:.1f}s | Vars: {len(prob.variables())}")

    # Solve
    solver = _make_solver(SOLVER_TIME_LIMIT)
    prob.solve(solver)

    t_solve = time.perf_counter()
    print(f"[TIMER] Solver: {t_solve - t_build:.1f}s | Status: {pulp.LpStatus[prob.status]}")

    # ============================================================
    # KẾT QUẢ (Tạm thời)
    # ============================================================
    print("=== HOÀN THÀNH V4.2 ===")
    excel_buffer = io.BytesIO()
    # TODO: Thêm phần ghi Excel sau khi test ổn
    return excel_buffer, 0, 0

# ============================================================
    # 4. KẾT QUẢ VÀ GÁN CONTAINER
    # ============================================================
    result_rows = []
    for (h, s, bay, b), yvar in y_vars.items():
        yv = pulp.value(yvar)
        if yv is not None and yv > 0.5:
            for dkey in demands[(h, s, bay)]:
                xkey = (h, s, bay, b, dkey)
                if xkey not in x_vars: continue
                qty = pulp.value(x_vars[xkey])
                if qty is not None and qty > 0.5:
                    w, st_v, pod_v = dkey
                    result_rows.append({
                        'MOVE HOUR': h, 'STS': s, 'BAY': bay,
                        'ASSIGNED BLOCK': b, 'WEIGHT CLASS': w,
                        'ST': st_v, 'POD': pod_v, 'QUANTITIES': int(round(qty))
                    })
    df_result = pd.DataFrame(result_rows)
    df_result['_sort_hr'] = df_result['MOVE HOUR'].map(hour_rank)
    df_result.sort_values(['_sort_hr','STS','BAY','ASSIGNED BLOCK'], inplace=True)
    df_result.drop(columns=['_sort_hr'], inplace=True)

    df_result_detail = []
    if container_data_available:
        # Build pool
        pool = defaultdict(list)
        for row in df_containers[['YARD','YB','YR','YT','REAL_WC',
                                   'YARD_POS','REAL_CONT_ID',
                                   'CONT_ST','CONT_POD']].itertuples(index=False):
            c = {'yb': int(row.YB), 'yr': int(row.YR), 'yt': int(row.YT),
                 'wc': int(row.REAL_WC), 'yard_pos': row.YARD_POS,
                 'real_cont_id': row.REAL_CONT_ID,
                 'st': row.CONT_ST, 'pod': row.CONT_POD,
                 'picked': False, 'pick_h': None}
            pool[row.YARD].append(c)

        # ── O(1) accessibility: pre-compute blocked_count và below_map ──
        blocked_count = {}  # id(cont) → int (số containers phía trên chưa pick)
        below_map     = {}  # id(cont) → [containers bên dưới nó]
        for blk, conts in pool.items():
            stacks = defaultdict(list)
            for c in conts:
                stacks[(c['yb'], c['yr'])].append(c)
            for c in conts:
                above = [o for o in stacks[(c['yb'], c['yr'])]
                         if o is not c and o['yt'] > c['yt']]
                below = [o for o in stacks[(c['yb'], c['yr'])]
                         if o is not c and o['yt'] < c['yt']]
                blocked_count[id(c)] = len(above)
                below_map[id(c)]     = below

        # avail[(block, wc, st, pod)] = list of currently accessible containers
        avail = defaultdict(list)
        for blk, conts in pool.items():
            for c in conts:
                if blocked_count[id(c)] == 0:
                    avail[(blk, c['wc'], c['st'], c['pod'])].append(c)

        opened_ybs = set()

        def pick_n(block, wc, st_match, pod_match, qty, h, s_job, bay_job, h_rank_val, result_list):
            remaining = qty
            akey = (block, wc, st_match, pod_match)
            av = avail[akey]
            while remaining > 0:
                if not av:
                    break
                # Chọn container tốt nhất
                yb_cnt = defaultdict(int)
                for c in av: yb_cnt[c['yb']] += 1
                best_i = min(range(len(av)), key=lambda i: (
                    0 if (block, av[i]['yb']) in opened_ybs else 1,
                    -yb_cnt[av[i]['yb']],
                    av[i]['yb'], av[i]['yr'], -av[i]['yt']
                ))
                best = av.pop(best_i)
                best['picked'] = True; best['pick_h'] = h
                opened_ybs.add((block, best['yb']))

                # Cập nhật blocked_count O(1): chỉ duyệt containers bên dưới best
                for lower in below_map.get(id(best), []):
                    if lower['picked']: continue
                    blocked_count[id(lower)] -= 1
                    if blocked_count[id(lower)] == 0:
                        # Container này vừa trở nên accessible
                        avail[(block, lower['wc'], lower['st'], lower['pod'])].append(lower)

                result_list.append({
                    'MOVE HOUR': h, 'CONTAINER ID': best['real_cont_id'],
                    'ST': best['st'], 'POD': best['pod'],
                    'STS': s_job, 'BAY': bay_job,
                    'ASSIGNED BLOCK': block, 'WEIGHT CLASS': wc,
                    'QUANTITIES': qty,
                    'YB': best['yb'], 'YR': best['yr'], 'YT': best['yt'],
                    'YARD POSITION': best['yard_pos']
                })
                remaining -= 1
            return remaining

        df_result_sorted = df_result.copy()
        df_result_sorted['_hr'] = df_result_sorted['MOVE HOUR'].map(hour_rank)
        df_result_sorted.sort_values(['_hr','STS','BAY','ASSIGNED BLOCK','WEIGHT CLASS'], inplace=True)

        deferred = []
        for h in all_hours_sorted:
            h_rank_val = hour_rank[h]
            for _, asg in df_result_sorted[df_result_sorted['MOVE HOUR'] == h].iterrows():
                s, bay_job, b = asg['STS'], asg['BAY'], asg['ASSIGNED BLOCK']
                w = int(asg['WEIGHT CLASS'])
                st_v  = str(asg.get('ST','')).strip()
                pod_v = str(asg.get('POD','')).strip()
                qty   = int(asg['QUANTITIES'])
                if b not in pool:
                    df_result_detail.append({
                        'MOVE HOUR': h,'STS': s,'BAY': bay_job,
                        'ASSIGNED BLOCK': b,'WEIGHT CLASS': w,
                        'CONTAINER ID':'','ST':'','POD':'','QUANTITIES': qty,
                        'YB':'','YR':'','YT':'','YARD POSITION':''
                    })
                    continue
                rem = pick_n(b, w, st_v, pod_v, qty, h, s, bay_job, h_rank_val, df_result_detail)
                if rem > 0:
                    deferred.append({'b':b,'wc':w,'st':st_v,'pod':pod_v,'qty':rem,
                                     'h_orig':h,'s':s,'bay':bay_job,'h_rank_min':h_rank_val})
            still_def = []
            for d in deferred:
                rem = pick_n(d['b'],d['wc'],d['st'],d['pod'],d['qty'],
                             h, d['s'],d['bay'],h_rank_val, df_result_detail)
                if rem > 0:
                    d2 = d.copy(); d2['qty'] = rem; still_def.append(d2)
            deferred = still_def

        rh = sum(d['qty'] for d in deferred)
        if rh > 0:
            print(f"  Re-handling: {rh} containers.")
            for d in deferred:
                print(f"    Block {d['b']} WC{d['wc']} x{d['qty']} (from {d['h_orig']})")
        else:
            print("  All containers assigned – no re-handling.")

        df_result_detail = pd.DataFrame(df_result_detail)
    else:
        df_result_detail = df_result.copy()
        df_result_detail.insert(1,'CONTAINER ID','')
        df_result_detail.insert(2,'ST','')
        df_result_detail.insert(3,'POD','')
        df_result_detail['YB'] = ''
        df_result_detail['YR'] = ''
        df_result_detail['YT'] = ''
        df_result_detail['YARD POSITION'] = ''

    df_result_detail['_sort_hr'] = df_result_detail['MOVE HOUR'].map(hour_rank)
    df_result_detail.sort_values(
        ['_sort_hr','STS','BAY','ASSIGNED BLOCK','WEIGHT CLASS','YB','YR','YT'],
        inplace=True)
    df_result_detail.drop(columns=['_sort_hr'], inplace=True)

    t_assign = time.perf_counter()
    print(f"[TIMER] Gán containers: {t_assign - t_solve:.1f}s")

    # ============================================================
    # 5. CLASH
    # ============================================================
    clash_details = []
    total_clashes = 0
    for (h, b), evar in e_vars.items():
        e_val = pulp.value(evar)
        if e_val is not None and e_val > 0.5:
            total_clashes += e_val
            u_val = pulp.value(u_vars[(h, b)])
            jobs = [f"{s}@{bay}" for (s, bay) in jobs_by_hour.get(h, [])
                    if pulp.value(y_vars.get((h, s, bay, b))) is not None
                    and pulp.value(y_vars.get((h, s, bay, b))) > 0.5]
            clash_details.append({
                'MOVE HOUR': h, 'BLOCK': b,
                'SỐ LƯỢNG BAY (u)': int(u_val) if u_val else 0,
                'CLASH (e = u-1)': int(e_val),
                'DANH SÁCH JOB (STS@BAY)': ', '.join(jobs)
            })
    df_clash = pd.DataFrame(clash_details)
    if not df_clash.empty:
        df_clash['_sort_hr'] = df_clash['MOVE HOUR'].map(hour_rank)
        df_clash.sort_values(['_sort_hr','BLOCK'], inplace=True)
        df_clash.drop(columns=['_sort_hr'], inplace=True)
    print(f"Total clashes: {total_clashes}")

    # ============================================================
    # 6. GHI EXCEL
    # ============================================================
    import openpyxl
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    _border = _thin_border()

    # CLASH sheet
    ws_clash = wb.create_sheet('CLASH')
    headers_clash = ['MOVE HOUR','BLOCK','SỐ LƯỢNG BAY (u)','CLASH (e = u-1)','DANH SÁCH JOB (STS@BAY)']
    for c_idx, hdr in enumerate(headers_clash, 1):
        cell = ws_clash.cell(row=1, column=c_idx, value=hdr)
        cell.font = _font(bold=True, color=C_WHITE)
        cell.fill = _fill(C_DARK_BLUE)
        cell.alignment = _align(); cell.border = _border
    if not df_clash.empty:
        for r_idx, row in enumerate(df_clash.itertuples(index=False), 2):
            for c_idx, val in enumerate(row, 1):
                cell = ws_clash.cell(row=r_idx, column=c_idx, value=val)
                cell.font = _font()
                cell.fill = _fill(C_WHITE if r_idx % 2 == 0 else C_ALT_ROW)
                cell.alignment = _align(); cell.border = _border
    else:
        cell = ws_clash.cell(row=2, column=1, value='Không có clash nào xảy ra.')
        cell.font = _font(); cell.fill = _fill(C_WHITE); cell.alignment = _align()
        ws_clash.merge_cells(start_row=2, start_column=1, end_row=2, end_column=5)
    for col, w in zip('ABCDE', [14,12,18,18,50]):
        ws_clash.column_dimensions[col].width = w

    # RESULT columns
    core_cols     = ['MOVE HOUR','CONT LIST','CONTAINER ID','ST','POD','STS','BAY',
                     'ASSIGNED BLOCK','WEIGHT CLASS','QUANTITIES']
    position_cols = ['YB','YR','YT','YARD POSITION']
    all_result_cols = (core_cols + position_cols) if container_data_available else \
                      ['MOVE HOUR','STS','BAY','ASSIGNED BLOCK','WEIGHT CLASS','QUANTITIES']
    CONT_LIST_COLS = {'CONT LIST'}; CONT_ID_COLS = {'CONTAINER ID','ST','POD'}
    POSITION_COLS  = set(position_cols)
    col_widths = {'MOVE HOUR':14,'CONT LIST':45,'CONTAINER ID':20,'ST':10,'POD':10,
                  'STS':10,'BAY':10,'ASSIGNED BLOCK':16,'WEIGHT CLASS':14,'QUANTITIES':12,
                  'YB':8,'YR':8,'YT':8,'YARD POSITION':18}

    # Pre-create style objects (dùng lại thay vì tạo mới mỗi cell)
    _fw   = _font(color='FF000000')
    _fw9  = _font(color='FF000000', size=9)
    _fwb  = _font(bold=True, color=C_WHITE)
    _ac   = _align()
    _atl  = Alignment(horizontal='left', vertical='top', wrap_text=True)
    _aw   = _align(wrap=True)
    _fp   = _fill(C_PALE_BLUE)
    _falt = _fill(C_ALT_ROW)
    _fw2  = _fill(C_WHITE)

    def write_result_sheet(ws, df, sheet_title):
        n_rows = len(df)
        cont_list_map = {}
        if container_data_available and 'CONTAINER ID' in df.columns:
            for (mh, bay), grp in df.groupby(['MOVE HOUR','BAY']):
                ids = [str(v).strip() for v in grp['CONTAINER ID']
                       if str(v).strip() not in ('','nan')]
                cont_list_map[(mh, bay)] = ', '.join(ids) if ids else ''

        for c_idx, cn in enumerate(all_result_cols, 1):
            cell = ws.cell(row=1, column=c_idx, value=cn)
            cell.fill = _fill(C_PALE_BLUE  if cn in CONT_LIST_COLS else
                              C_MID_BLUE   if cn in CONT_ID_COLS   else
                              C_LIGHT_BLUE if cn in POSITION_COLS  else C_DARK_BLUE)
            cell.font = _fwb; cell.alignment = _aw; cell.border = _border

        merge_groups = []
        cl_col = all_result_cols.index('CONT LIST') + 1 if 'CONT LIST' in all_result_cols else None
        if container_data_available and cl_col:
            prev_key = None; grp_start = 2
            for i, (_, row) in enumerate(df.iterrows()):
                cur_key = (row.get('MOVE HOUR',''), row.get('BAY',''))
                er = i + 2
                if cur_key != prev_key:
                    if prev_key is not None:
                        merge_groups.append((prev_key, grp_start, er-1, cont_list_map.get(prev_key,'')))
                    prev_key = cur_key; grp_start = er
            if prev_key is not None:
                merge_groups.append((prev_key, grp_start, n_rows+1, cont_list_map.get(prev_key,'')))

        group_key = None; group_shade = C_ALT_ROW
        for r_idx, (_, row) in enumerate(df.iterrows(), 2):
            this_key = (row.get('MOVE HOUR'), row.get('STS'), row.get('BAY'),
                        row.get('ASSIGNED BLOCK'), row.get('WEIGHT CLASS'))
            if this_key != group_key:
                group_shade = C_WHITE if group_shade == C_ALT_ROW else C_ALT_ROW
                group_key = this_key
            cur_fill = _fw2 if group_shade == C_WHITE else _falt
            for c_idx, cn in enumerate(all_result_cols, 1):
                val = None if cn == 'CONT LIST' else row.get(cn,'')
                if cn in ('YB','YR','YT') and val != '':
                    try: val = int(val)
                    except: pass
                if val == '' or (isinstance(val, float) and str(val) == 'nan'):
                    val = None
                cell = ws.cell(row=r_idx, column=c_idx, value=val)
                cell.font = _fw; cell.fill = cur_fill
                cell.alignment = _ac; cell.border = _border

        if cl_col:
            for (mh, bay), r_start, r_end, list_text in merge_groups:
                cell = ws.cell(row=r_start, column=cl_col, value=list_text or None)
                cell.font = _fw9; cell.fill = _fp
                cell.alignment = _atl; cell.border = _border
                if r_end > r_start:
                    ws.merge_cells(start_row=r_start, start_column=cl_col,
                                   end_row=r_end, end_column=cl_col)
                    ws.cell(row=r_start, column=cl_col).alignment = _atl
            for (mh, bay), r_start, r_end, list_text in merge_groups:
                span = r_end - r_start + 1
                n_ids = len([x for x in list_text.split(',') if x.strip()]) if list_text else 0
                rh = max(15, min(60, max(1, -(-n_ids // max(1, span))) * 13))
                for r in range(r_start, r_end+1):
                    ws.row_dimensions[r].height = rh

        for c_idx, cn in enumerate(all_result_cols, 1):
            ws.column_dimensions[get_column_letter(c_idx)].width = col_widths.get(cn, 14)
        print(f"  Sheet '{sheet_title}': {n_rows} rows written.")

    # RESULT sheets theo ST
    if container_data_available and 'ST' in df_result_detail.columns:
        st_values = [s for s in sorted(df_result_detail['ST'].dropna().unique())
                     if str(s).strip() not in ('','nan')]
    else:
        st_values = ['ALL']
    if not st_values: st_values = ['ALL']

    for st_idx, st_val in enumerate(st_values, 1):
        sname = (f"RESULT {st_idx} ({st_val})" if st_val != 'ALL' else 'RESULT')[:31]
        ws = wb.create_sheet(sname)
        df_rd = (df_result_detail if st_val == 'ALL' else
                 df_result_detail[df_result_detail['ST'].astype(str).str.strip() == str(st_val).strip()]
                 ).reset_index(drop=True)
        write_result_sheet(ws, df_rd, sname)

    ws_total = wb.create_sheet('RESULT TOTAL')
    write_result_sheet(ws_total, df_result_detail.reset_index(drop=True), 'RESULT TOTAL')

    # ============================================================
    # 7. LƯU BUFFER
    # ============================================================
    excel_buffer = io.BytesIO()
    wb.save(excel_buffer); excel_buffer.seek(0)
    t_end = time.perf_counter()
    total_rows = len(df_result_detail)
    print(f"[TIMER] Ghi Excel: {t_end - t_assign:.1f}s")
    print(f"[TIMER] ══ TỔNG CỘNG: {t_end - t0:.1f}s ══")
    print(f"Done. Rows={total_rows}, Total Clashes={total_clashes}")
    return excel_buffer, total_rows, total_clashes
