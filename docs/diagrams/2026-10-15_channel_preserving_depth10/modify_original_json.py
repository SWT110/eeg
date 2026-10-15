"""Modify the user's original diagram in its native schema, preserving its style.

Produces a sibling JSON and a two-tab comparison workspace; never overwrites
the original JSON or changes the HTML editor/model implementation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

EEG = Path(__file__).resolve().parents[3]
FOLDER = EEG / 'html_json/eeg_json'
SOURCE = FOLDER / 'activity_loso_time_fft_branchloss_crossdepthqkv_11_10_8_seed42_结构框架图.json'
OUTPUT = FOLDER / 'activity_loso_time_fft_channelpreserve_depth10_skip10_seed42_结构框架图.json'
COMPARE = FOLDER / 'activity_loso_time_fft_原图与depth10修改版_对比工作区.json'


def main():
    source_bytes = SOURCE.read_bytes()
    original = json.loads(source_bytes)
    d = copy.deepcopy(original)
    source_nodes = {n['id']: n for n in original['nodes']}
    nodes = {n['id']: n for n in d['nodes']}
    notes = {n['id']: n for n in d['notes']}
    edge_num = 0

    def edit(id, **changes):
        nodes[id].update(changes)

    def add(id, template='ov_time_cross', **changes):
        n = copy.deepcopy(source_nodes[template])
        n.update(id=id, **changes)
        d['nodes'].append(n)
        nodes[id] = n
        return id

    def note(id, text, **changes):
        if id not in notes:
            n = dict(id=id, x=4000, y=0, text=text, fontSize=18,
                     color='#555d68', parentId=None, anchor='free', dx=0, dy=0)
            d['notes'].append(n)
            notes[id] = n
        notes[id].update(text=text, **changes)

    def remove(ids):
        ids = set(ids)
        d['nodes'] = [n for n in d['nodes'] if n['id'] not in ids]
        d['notes'] = [n for n in d['notes'] if n['id'] not in ids]
        for id in ids:
            nodes.pop(id, None)
            notes.pop(id, None)

    def wire(a, b, ap='right', bp='left', route='auto', color='#343a40',
             arrow=True, bulge=100, width=1.8):
        nonlocal edge_num
        edge_num += 1
        e = dict(id=f'v2_e{edge_num:03d}',
                 **{'from': dict(nodeId=a, port=ap) if isinstance(a,str) else a,
                    'to': dict(nodeId=b, port=bp) if isinstance(b,str) else b},
                 route=route, arrow=arrow, bulge=bulge,
                 corner=0 if route=='straight' else 12, width=width, color=color)
        d['edges'].append(e)
        return e['id']

    blue, purple, orange = '#4776b4', '#9a5f9e', '#b27318'
    changed_fill = '#fff0da'

    # Keep all five original panels, the detailed Block, and the native palette,
    # settings, profiles and 8000x4500 viewport. Remove obsolete model elements.
    removed = {
        'ov_time_enc_11','ov_time_enc_8','ov_fft_enc_11','ov_fft_enc_8',
        'ov_time_cross','ov_fft_cross','fft_global_off','fft_input_qkv_off',
        'ov_concat_11','ov_concat_8','ov_head_11','ov_head_8',
        'ov_branch_11','ov_branch_8','ov_softmax_weights','ov_fused_logits',
        'ov_fused_features','loss_z_stack','loss_alpha','loss_softmax',
        'loss_fused_logits','loss_f_stack','loss_fused_features','loss_l_stack',
        'loss_weighted','loss_aux','loss_add',
    }
    removed |= {id for id in nodes if id.startswith(
        ('et11_','et8_','ef11_','ef8_','cd_','t11_','h11_','h8_'))}
    remove(removed)
    # Retain the original edge objects for the unchanged Block, the two Stem
    # chains (now containing explicit Identity stages), and the one retained MLP.
    def retained_edge(e):
        a, b = e['from'].get('nodeId',''), e['to'].get('nodeId','')
        if a not in nodes or b not in nodes:
            return False
        return (a.startswith(('blk_','ff_')) and b.startswith(('blk_','ff_'))
                or a.startswith('pt_') and b.startswith('pt_')
                or a.startswith('pf_') and b.startswith('pf_')
                or a.startswith('h10_') and b.startswith('h10_'))
    d['edges'] = [e for e in d['edges'] if retained_edge(e)]

    note('title','超详细结构框图：Time + FFT 保留电极通道 EEG-Conformer + 单深度10层 + 前后5层双段跳接')
    note('subtitle','基于原 11/10/8 JSON 修改 | window=15s, stride=3s | depth=10 (5+5) | skip to Block #1-#10 ends | seed=42 | 正式15人；真实训练待执行')
    note('overview_title','① 端到端总流程（保留原图布局；详细模块仍在下方对应分区展开）')
    note('overview_time','Time 域：单一 depth=10',x=1710,y=178)
    note('overview_fft','FFT 域：单一 depth=10',x=1710,y=570)
    for domain, y, n, letter, color in [('time',300,122,'T',blue),('fft',665,58,'F',purple)]:
        domain_label = 'Time' if domain=='time' else 'FFT'
        if domain=='time':
            edit('time_zscore',y=y+10)
            edit('time_input_qkv_off',y=y+10)
            edit('time_patch_summary',x=1120,y=y,w=365,h=106,
                 label='Time PatchEmbedding -> (B,21,122,40)')
            wire('raw_window','time_zscore',color=color)
            wire('time_zscore','time_input_qkv_off',route='straight',color=color)
            wire('time_input_qkv_off','time_patch_summary',color=color)
        else:
            edit('fft_patch_summary',x=1120,y=y,w=365,h=106,
                 label='FFT PatchEmbedding -> (B,21,58,40)')
            wire('raw_window','fft_transform',color=color)
            wire('fft_transform','fft_zscore',route='straight',color=color)
            wire('fft_zscore','fft_patch_summary',color=color)
        prefix = 'ov_'+domain
        edit(prefix+'_enc_10',x=1540,y=y,w=380,h=106,
             label=f'{domain_label} Blocks #1-#5 (per electrode)')
        add(prefix+'_channel_fuse',x=1980,y=y,w=350,h=106,
            label='Channel Conv 21->1 + squeeze',fill=changed_fill)
        add(prefix+'_late',template=prefix+'_enc_10',x=2390,y=y,w=350,h=106,
            label=f'{domain_label} Blocks #6-#10 (each + F5)')
        edit(prefix+'_updated',x=2800,y=y,w=320,h=106,label=f'{letter}10 (B,{n},40)')
        edit(prefix+'_flat',x=3180,y=y,w=355,h=106,
             label=f'Flatten {domain_label} -> (B,{n*40})')
        add(prefix+'_channel_skip',x=1540,y=y-102,w=790,h=65,
            label='Cross-electrode Conv 21->21 | skip to outputs of Blocks #1-#5',
            fill=changed_fill)
        wire(domain+'_patch_summary',prefix+'_enc_10',route='straight',color=color)
        fused_main_edge = None
        for a,b in [('enc_10','channel_fuse'),('channel_fuse','late'),
                    ('late','updated'),('updated','flat')]:
            edge_id = wire(prefix+'_'+a,prefix+'_'+b,route='straight',color=color)
            if a=='channel_fuse':
                fused_main_edge = edge_id
        wire(domain+'_patch_summary',prefix+'_channel_skip',ap='top',color=orange)
        wire(prefix+'_channel_skip',prefix+'_enc_10',ap='bottom',bp='top',route='turn-left-down',color=orange)
        wire(dict(edgeId=fused_main_edge,anchor='mid'),prefix+'_late',bp='top',
             route='arch-top',bulge=60,color=orange,width=2)
        note(prefix+'_late_skip_note','F5 -> each Block #6-#10 output',x=2650,y=y-90,
             color=orange,fontSize=16)
        note(prefix+'_early_shape',f'Before fusion: (B,21,{n},40)',x=1730,y=y+145,
             color=color,fontSize=16)

    edit('ov_concat_10',label='Concat Time + FFT -> f(B,7200)')
    edit('ov_head_10',label='Single Head: 7200->256->32->3')
    edit('ov_branch_10',label='logits z (B,3)')
    edit('ov_prediction',x=4850,y=255,w=1030,h=98,label='argmax(z) -> e_1 / e_2 / e_3')
    edit('ov_total_loss',x=4850,y=660,w=530,h=105,label='Ltotal = CE(z,y) | no depth auxiliary loss')
    edit('ov_backprop',x=5440,y=665,w=440,h=95,label='Adam + Backprop all modules')
    for p,c in [('time',blue),('fft',purple)]: wire('ov_'+p+'_flat','ov_concat_10',color=c)
    wire('ov_concat_10','ov_head_10',route='straight')
    wire('ov_head_10','ov_branch_10',route='straight')
    wire('ov_branch_10','ov_prediction',color='#c04b54')
    wire('ov_branch_10','ov_total_loss',color='#c04b54')
    wire('ov_total_loss','ov_backprop',route='straight',color='#c04b54')
    edit('cfg_depth',label='depth=10 (5+5) | skip to all 10 Block ends | cross_depth=off')
    edit('cfg_drop',label='Dropout: stem=.5 | encoder=.5 | head=.5/.3 | baseline defaults')
    edit('cfg_init',label='C=21 EEG electrodes | N=122/58 tokens | D=40 learned features',fill=changed_fill)
    edit('cfg_loss',label='single CE | depth aux=0 | class weights=None | seed=42')
    note('overview_config_title','已确认本地基线默认参数；depth=10 按录音单独覆盖')
    note('cfg_result_note','新方案未训练；旧多深度权重 w、CrossDepth gamma 和旧结果不适用于本图。',fontSize=16)
    note('cfg_disabled_note','fft_global=none | input_qkv=none | cumulative_query_attention=false | 无类别加权、无重采样、无 S&R')

    # PatchEmbedding: retain old boxes/locations as explicit Identity stages so
    # a tab switch exposes precisely where early spatial fusion was removed.
    note('patch_title','② PatchEmbedding 全步骤（Time / FFT 参数独立；电极维 C=21 全程保留）')
    note('patch_time_title','Time PatchEmbedding: (B,1,21,1920) -> (B,21,122,40)')
    note('patch_fft_title','FFT PatchEmbedding: (B,1,21,961) -> (B,21,58,40)')
    for p,n,length in [('pt',122,1896),('pf',58,937)]:
        edit(p+'_conv_s',label='Identity: keep 21 electrodes (defer spatial fusion)',fill=changed_fill)
        edit(p+'_squeeze',label='Identity: no squeeze',fill=changed_fill)
        edit(p+'_transpose',label='Permute -> (B,C,N,D)')
        note(p+'_conv_s_shape',f'(B,40,21,{length})')
        for suffix in ['bn','elu']:note(p+'_'+suffix+'_shape',f'(B,40,21,{length})')
        for suffix in ['pool','drop','proj','squeeze']:note(p+'_'+suffix+'_shape',f'(B,40,21,{n})')
        note(p+'_transpose_shape',f'(B,21,{n},40)')
    note('patch_time_change','原 k=(21,1) 空间卷积移出 Stem；第 5 个 EncoderBlock 之后再融合电极。',x=2010,y=1375,fontSize=20,color=orange)
    note('patch_fft_change','保留原图位置的 Identity 框便于对比；它们不代表新增可学习模块。',x=5990,y=1375,fontSize=20,color=orange)
    note('patch_formula_time','Time patch count: ((1920-24)-75)//15 + 1 = 122 | C=21 remains | D=40 is feature size')
    note('patch_formula_fft','FFT patch count: ((961-24)-75)//15 + 1 = 58 | C=21 remains | independent stem parameters')
    note('patch_preprocess_note','FFT = log1p(abs(rFFT(x))^2); train-only standardization unchanged | 删除 squeeze 本身不能恢复已被卷积融合的电极维')

    # The original third-row left inset becomes two single-depth encoders.
    note('transformer_title','③ 两个独立的10层 Encoder（各域5+5）+ 单个 TransformerEncoderBlock 完整内部流程')
    note('encoder_repeats_title','Time / FFT 各一条10层链；同域各电极共享每层 Block 参数')
    for p,source,y,n,letter,color in [('et','enc_time_source',1950,122,'T',blue),
                                    ('ef','enc_fft_source',2280,58,'F',purple)]:
        edit(source,y=y,w=285,h=100,label=f'E_{"time" if p=="et" else "fft"} (B,21,{n},40)')
        edit(p+'10_b1',x=430,y=y,w=430,h=100,label='Blocks #1-#5 per electrode')
        edit(p+'10_b2',x=915,y=y,w=415,h=100,label='Channel Conv 21->1 + squeeze',fill=changed_fill)
        edit(p+'10_mid',x=1385,y=y,w=350,h=100,label='Blocks #6-#9 (each + F5)')
        edit(p+'10_bn',x=1790,y=y,w=280,h=100,label='EncoderBlock #10 + F5')
        edit(p+'10_out',x=2125,y=y,w=500,h=100,label=f'{letter}10 (B,{n},40)')
        add(p+'10_skip',x=430,y=y-130,w=900,h=70,
            label='Input Conv 21->21 -> skip to outputs of Blocks #1-#5',fill=changed_fill)
        wire(source,p+'10_b1',route='straight',color=color)
        fused_main_edge = None
        for a,b in [('b1','b2'),('b2','mid'),('mid','bn'),('bn','out')]:
            edge_id = wire(p+'10_'+a,p+'10_'+b,route='straight',color=color)
            if a=='b2':
                fused_main_edge = edge_id
        wire(source,p+'10_skip',ap='top',color=orange)
        wire(p+'10_skip',p+'10_b1',ap='bottom',bp='top',route='turn-left-down',color=orange)
        for target in ['mid','bn']:
            wire(dict(edgeId=fused_main_edge,anchor='mid'),p+'10_'+target,bp='top',
                 route='arch-top',bulge=45,color=orange,width=2)
    note('encoder_repeat_note1','Time: F5 (B,122,40) -> skip to each Block #6-#10 output',x=1405,y=2090,fontSize=17,color=blue)
    note('encoder_repeat_note2','FFT: F5 (B,58,40) -> skip to each Block #6-#10 output',x=1405,y=2410,fontSize=17,color=purple)
    note('encoder_repeat_note3','共20个 Block；各层、Time/FFT 两域参数独立；同域各电极共享每层参数。',x=1405,y=2460,fontSize=15)
    edit('blk_x',label='Input (B_eff,N,40)')
    edit('ff_out',label='Block output (B_eff,N,40)')
    note('block_batch_note','前5层：B_eff=B×21，可 reshape 为 (B×21,N,40)；后5层：B_eff=B。Block 内部结构沿用基线。',x=5360,y=2480,fontSize=16)

    # Replace the old fourth-row CrossDepth QKV with two corresponding zooms.
    note('cross_title','④ 双段跳接：入口21->21卷积接前5层末尾；第5层后21->1融合特征接后5层末尾')
    note('cross_generic_title','入口旁路：每域各计算一次 S；同一 S 分别接入前5个完整 Block 的末尾；Time N=122，FFT N=58')
    add('skip_x0',template='cd_input',x=95,y=2825,w=440,h=90,label='X0 (B,21,N,40)')
    add('skip_conv',template='cd_qkv_proj',x=595,y=2705,w=550,h=80,
        label='Cross-electrode Conv C:21->21',fill=changed_fill)
    add('skip_s',template='cd_output',x=1205,y=2705,w=550,h=80,label='S (B,21,N,40)')
    wire('skip_x0','skip_conv',ap='top',color=orange)
    wire('skip_conv','skip_s',route='straight',color=orange)
    wire('skip_s',dict(x=6946,y=2745),route='straight',color=orange,arrow=False,width=2)
    for i in range(1,6):
        x=2070+(i-1)*1020
        add(f'skip_block_{i}',template='et10_b1',x=x,y=2830,w=670,h=80,
            label=f'Per-electrode EncoderBlock #{i}')
        add(f'skip_add_{i}',template='cd_resadd',x=x+750,y=2844,w=52,h=52,label='+',fill=changed_fill)
        wire('skip_x0' if i==1 else f'skip_add_{i-1}',f'skip_block_{i}',route='straight')
        wire(f'skip_block_{i}',f'skip_add_{i}',route='straight')
        wire(dict(x=x+776,y=2745),f'skip_add_{i}',bp='top',route='straight',color=orange,width=2)
    add('skip_h5',template='cd_output',x=7120,y=2830,w=650,h=80,label='H5 (B,21,N,40)')
    wire('skip_add_5','skip_h5',route='straight')
    note('cross_generic_note','S=Conv_C_21to21(X0); H_i=Block_i(H_(i-1))+S, i=1..5 | 固定同一 S，分别加到前5个完整 Block 的输出。',fontSize=18)
    note('cross_t11_title','第5层后：C=21 -> 1，再 squeeze 得到 F5；F5 作为第6层输入，并分别跳接到第6-10层末尾')
    late_nodes=[
        ('late_h5','cd_input',95,500,'H5 (B,21,N,40)',None),
        ('late_conv','cd_qkv_proj',655,630,'Cross-electrode Conv C:21->1',changed_fill),
        ('late_squeeze','cd_split',1345,580,'Squeeze -> F5 (B,N,40)',changed_fill),
    ]
    for id,template,x,w,label,fill in late_nodes:
        fields=dict(x=x,y=3240,w=w,h=90,label=label)
        if fill: fields['fill']=fill
        add(id,template,**fields)
    for a,b in zip(late_nodes,late_nodes[1:]):wire(a[0],b[0],route='straight',color=blue)
    late_main_edge = None
    for i in range(6,11):
        x=2130+(i-6)*1010
        add(f'late_block_{i}',template='et10_b1',x=x,y=3240,w=600,h=90,
            label=f'EncoderBlock #{i} (B,N,40)')
        add(f'late_add_{i}',template='cd_resadd',x=x+690,y=3259,w=52,h=52,
            label='+',fill=changed_fill)
        edge_id=wire('late_squeeze' if i==6 else f'late_add_{i-1}',
                     f'late_block_{i}',route='straight',color=blue)
        if i==6:
            late_main_edge=edge_id
        wire(f'late_block_{i}',f'late_add_{i}',route='straight',color=blue)
        wire(dict(x=x+716,y=3180),f'late_add_{i}',bp='top',
             route='straight',color=orange,width=2)
    # Branch after the one 21->1 convolution + squeeze; reuse that same F5.
    wire(dict(edgeId=late_main_edge,anchor='mid'),dict(x=2027.5,y=3180),
         route='straight',color=orange,arrow=False,width=2)
    wire(dict(x=2027.5,y=3180),dict(x=6886,y=3180),
         route='straight',color=orange,arrow=False,width=2)
    add('late_out',template='cd_output',x=7050,y=3240,w=750,h=90,
        label='Final output (B,N,40)')
    wire('late_add_10','late_out',route='straight',color=blue)
    add('late_flat',template='t11_flat',x=7050,y=3390,w=750,h=65,
        label='Flatten: Time (B,4880) | FFT (B,2320)')
    wire('late_out','late_flat',ap='bottom',bp='top',route='straight',color=blue)
    note('late_skip_bus_label','同一融合特征 F5 分别跳接到 Block #6、#7、#8、#9、#10 的末尾（两域各自计算）',
         x=4700,y=3155,fontSize=18,color=orange)
    note('late_skip_formula','F5=squeeze(Conv_C_21to1(H5)); Z5=F5; Z_i=Block_i(Z_(i-1))+F5, i=6..10',
         x=3500,y=3395,fontSize=18)
    note('t11_key_note','F5 在后5层复用；21->1融合卷积只在第5层后计算一次。21是电极数，40是特征维。',
         x=3500,y=3465,fontSize=20,color=orange)
    note('cross_scope_note','已确认 Conv2d(C,C,1) / Conv2d(C,1,1)，groups=1，含 bias；两段固定特征相加，Time/FFT 独立。',
         y=3550,fontSize=18)

    # Preserve the original middle MLP row verbatim except semantic labels/loss.
    note('headloss_title','⑤ 单一分类头、Time/FFT 特征拼接、单一分类损失与训练参数（不再有多深度权重）')
    note('heads_detail_title','保留原图中间一套 MLP Head：两域特征拼接后做三分类')
    note('loss_detail_title','预测直接使用单一 logits；训练直接对其计算分类交叉熵')
    edit('h10_tf',label='Flatten T10 (B,4880)')
    edit('h10_ff',label='Flatten F10 (B,2320)')
    edit('h10_cat',label='Concat f (B,7200)')
    edit('h10_z',label='logits z (B,3)')
    edit('h10_ce',label='CrossEntropy(z,y), class weights=None')
    edit('h10_loss',label='L scalar')
    add('loss_logits',template='loss_z_stack',x=4440,y=3830,w=550,h=90,label='Single logits z (B,3)')
    edit('loss_argmax',x=5050,y=3830,w=2760,h=90,label='argmax(z) -> e_1 / e_2 / e_3 (class 0 / 1 / 2)')
    edit('loss_total',x=4440,y=4130,w=1000,h=102,label='Ltotal = CE(z,y)')
    edit('loss_adam',x=5500,y=4115,w=2310,h=117,
         label='Adam: lr=.0002, betas=(.5,.999), eps=1e-8, weight_decay=0 | Backprop Stem + channel Conv + Encoders + Head')
    wire('h10_z','loss_logits',ap='top',bp='top',route='arch-top',bulge=50,color='#c04b54')
    wire('loss_logits','loss_argmax',route='straight',color='#c04b54')
    wire('h10_loss','loss_total',color='#c04b54')
    wire('loss_total','loss_adam',color='#c04b54')
    note('single_head_note','原三套 depth Head 改为单套；保留 7200->256->32->3、ELU、Dropout .5/.3。',x=2210,y=3870,fontSize=20,color=blue)
    note('single_loss_removed','移除 alpha / softmax 深度权重、weighted logits、三分支 CE 和 depth auxiliary loss。',x=6165,y=4020,fontSize=20,color=orange)
    note('head_loss_note','已确认单一无权重 CE、原始采样、无 S&R；S6 与 S17 单列，正式队列 15 人。',fontSize=19)
    note('training_note','epochs=200 | batch_size=72 | lr=.0002 | seed=42；训练入口 train_activity_channelpreserve.py；真实训练待服务器执行。',fontSize=17)

    # Schema and topology checks, preserving the user's original data exactly.
    ids = [n['id'] for n in d['nodes']]
    assert len(ids)==len(set(ids))
    edge_ids=[e['id'] for e in d['edges']]
    assert len(edge_ids)==len(set(edge_ids))
    note_ids=[n['id'] for n in d['notes']]
    assert len(note_ids)==len(set(note_ids))
    for e in d['edges']:
        for end in ('from','to'):
            ref=e[end]
            assert not ref.get('nodeId') or ref['nodeId'] in ids, e
            assert not ref.get('edgeId') or ref['edgeId'] in edge_ids, e
    for key in ('settings','profileGroups','palette','view'):
        assert d[key]==original[key]
    assert set(d)==set(original)
    assert all(n['x']>=0 and n['y']>=0 and n['x']+n['w']<=8000 and n['y']+n['h']<=4500 for n in d['nodes'])
    OUTPUT.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    workspace=dict(format='neural-diagram-workspace',version=4,
        activeDocumentId='channel_preserve_depth10_skip10',documents=[
            dict(id='original_11_10_8',title='原图：11-10-8 + CrossDepth QKV + Branch Loss',state=original,dirty=False),
            dict(id='channel_preserve_depth10_skip10',title='修改版：depth10 + 前后5层双段跳接',state=d,dirty=False)])
    COMPARE.write_text(json.dumps(workspace,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert SOURCE.read_bytes()==source_bytes
    print(json.dumps(dict(output=str(OUTPUT),comparison=str(COMPARE),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        nodes=len(d['nodes']),edges=len(d['edges']),notes=len(d['notes']),
        original_node_ids_retained=len(set(ids)&set(source_nodes))),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
