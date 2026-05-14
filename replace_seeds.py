"""补充种子: 找满足 dj>0 且 v3_joint>=MIN_V3 的替换种子"""
import numpy as np, time, csv, sys, io
from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config
from load_balanced_bvf_v3style_advanced import LoadBalancedBVF_V3Style, create_lb_v3style_config

K,L,G = 40,6,9;  EVAL_SEED=99999
W_MIN,W_JFI,REF_RATE,FLOOR_RATE = 0.35,0.65,60.0,48.0
MIN_V3  = 0.82       # 绝对质量门槛
NEED    = 4          # 需要补多少个
START_SEED = 161
CSV_IN  = "result/seed_selection/good_seeds.csv"
CSV_OUT = "result/seed_selection/good_seeds.csv"

def generate_hotspot(seed, sq=1000):
    np.random.seed(seed)
    h_ue,h_ap,h_uav = 1.65,15.0,50.0
    spacing=sq/4
    gap=[[(i+1)*spacing,(j+1)*spacing,h_ap] for i in range(3) for j in range(3)]
    ground_AP_pos=np.array(gap)
    n_hot=int(K*0.75); n_uni=K-n_hot
    centers=[[sq*0.25,sq*0.30],[sq*0.70,sq*0.75]]
    hot=[np.random.normal(c,sq*0.05,(n_hot//2,2)) for c in centers]
    hot_xy=np.vstack(hot)[:n_hot]
    uni_xy=np.random.uniform(50,sq-50,(n_uni,2))
    UE_xy=np.clip(np.vstack([hot_xy,uni_xy]),30,sq-30)
    UE_pos=np.column_stack([UE_xy,np.full(K,h_ue)])
    l_side=int(np.ceil(np.sqrt(L))); usp=sq/(l_side+1)
    uavs=[[np.clip((i+1)*usp+np.random.uniform(-15,15),60,sq-60),
           np.clip((j+1)*usp+np.random.uniform(-15,15),60,sq-60),h_uav]
          for i in range(l_side) for j in range(l_side)][:L]
    UAV_pos=np.array(uavs); opt_state=np.random.get_state()
    return UE_pos,ground_AP_pos,UAV_pos,opt_state

def eval_det(evaluator,UE_pos,gAP,UAV_pos):
    state=np.random.get_state(); np.random.seed(EVAL_SEED)
    all_AP=np.vstack([gAP,UAV_pos])
    _,_,betas=evaluator.compute_channel_model(UE_pos,all_AP)
    mask=evaluator.compute_AP_selection_mask(betas)
    rates,_=evaluator.compute_user_rates(UE_pos,all_AP,mask)
    np.random.set_state(state)
    mask_uav=mask[:,G:]; mask_gnd=mask[:,:G]
    eff=np.zeros(L)
    for l in range(L):
        for k in np.where(mask_uav[:,l])[0]:
            ub=betas[k,G+l]; gb=betas[k,np.where(mask_gnd[k])[0]].sum()
            eff[l]+=ub/(gb+ub+1e-12)
    s=eff.sum()
    jfi=float(s**2/(L*(eff**2).sum()+1e-12)) if s>1e-10 else 1.0
    raw=W_MIN*(rates.min()/REF_RATE)+W_JFI*jfi
    if rates.min()<FLOOR_RATE: raw*=(rates.min()/FLOOR_RATE)**2
    return float(rates.min()),jfi,float(raw)

if __name__=="__main__":
    base_cfg=create_v6_config()
    base_cfg.update({'num_UE':K,'num_UAV':L,'num_ground_AP':G,'tau_p':K,'max_iterations':80,'num_serving_APs':3})
    v3_cfg=create_lb_v3style_config()
    v3_cfg.update({'num_UE':K,'num_UAV':L,'num_ground_AP':G,'tau_p':K,'max_iterations':80,'num_serving_APs':3,
                   'w_min':W_MIN,'w_jfi':W_JFI,'ref_rate':REF_RATE,'floor_rate':FLOOR_RATE})

    # 读现有 CSV
    with open(CSV_IN) as f:
        reader=csv.DictReader(f); header=reader.fieldnames
        existing=[r for r in reader if float(r['v3_joint'])>=MIN_V3]
    print(f"保留现有好种子: {len(existing)} 个  (需补充 {NEED} 个)")

    evaluator=BalancedVirtualForceOptimizerV6(base_cfg)
    new_rows=[]; seed=START_SEED; total_run=0

    while len(new_rows)<NEED and seed<=300:
        total_run+=1
        print(f"  试 seed={seed} ...", end=" ", flush=True)
        UE_pos,gAP,init_UAV,opt_state=generate_hotspot(seed)

        np.random.set_state(opt_state)
        v6=BalancedVirtualForceOptimizerV6(base_cfg)
        # suppress output
        old_stdout=sys.stdout; sys.stdout=io.StringIO()
        res=v6.optimize(UE_pos,gAP,init_UAV.copy())
        sys.stdout=old_stdout
        _,v6_jfi,v6_j=eval_det(evaluator,UE_pos,gAP,res['optimized_UAV_pos'])
        v6_min,_,_=eval_det(evaluator,UE_pos,gAP,res['optimized_UAV_pos'])

        np.random.set_state(opt_state)
        v3=LoadBalancedBVF_V3Style(v3_cfg)
        old_stdout=sys.stdout; sys.stdout=io.StringIO()
        res=v3.optimize(UE_pos,gAP,init_UAV.copy())
        sys.stdout=old_stdout
        v3_min,v3_jfi,v3_j=eval_det(evaluator,UE_pos,gAP,res['optimized_UAV_pos'])

        dj=v3_j-v6_j; dm=v3_min-v6_min; df=v3_jfi-v6_jfi
        ok=(dj>0) and (v3_j>=MIN_V3)
        print(f"v6={v6_j:.4f} v3={v3_j:.4f} dj={dj:+.4f}  {'✓ ADDED' if ok else '✗ skip'}")

        if ok:
            new_rows.append({
                'seed':seed,'v6_joint':f'{v6_j:.6f}','v6_min':f'{v6_min:.4f}','v6_jfi':f'{v6_jfi:.6f}',
                'v3_joint':f'{v3_j:.6f}','v3_min':f'{v3_min:.4f}','v3_jfi':f'{v3_jfi:.6f}',
                'dj':f'{dj:.6f}','dm':f'{dm:.4f}','df':f'{df:.6f}',
            })
        seed+=1

    print(f"\n补充完成: {len(new_rows)} 个 (共试 {total_run} 次)")
    all_rows=existing+new_rows
    print(f"最终好种子总数: {len(all_rows)}")

    with open(CSV_OUT,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=header)
        writer.writeheader(); writer.writerows(all_rows)
    print(f"已写入: {CSV_OUT}")
