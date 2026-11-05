"""Draw the proposed architecture; does not modify or train the EEG model.

The canonical vector scene exports to SVG, PNG, draw.io, and the repository's
existing neural-network-diagram-editor JSON format. Requires Pillow only.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONT = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
if not FONT.exists():
    FONT = Path('/Library/Fonts/Arial Unicode.ttf')

INK = '#243449'
MUTED = '#53677c'
LINE = '#6c8299'
BORDER = '#cbd5e1'
BLUE = '#e6f0fb'
PURPLE = '#f0e9f8'
GREEN = '#e7f0dd'
TEAL = '#e1f0ed'
ORANGE = '#fff0da'
ORANGE_INK = '#a25d14'
ROSE = '#f8e6e8'
GREY = '#f3f5f8'


class Scene:
    def __init__(self, width, height, title):
        self.w, self.h, self.title = width, height, title
        self.items = []
        self.boxes = {}
        self.errors = []

    def text(self, x, y, text, size=23, color=INK, anchor='middle', bold=False):
        self.items.append(dict(kind='text', x=x, y=y, text=text, size=size,
                               color=color, anchor=anchor, bold=bold))

    def box(self, id, x, y, w, h, lines=(), fill=BLUE, size=23,
            stroke=BORDER, radius=10, color=INK, kind='box'):
        obj = dict(kind=kind, id=id, x=x, y=y, w=w, h=h, fill=fill,
                   stroke=stroke, radius=radius)
        self.items.append(obj)
        self.boxes[id] = obj
        if isinstance(lines, str):
            lines = lines.split('\n')
        gap = size * 1.4
        for i, line in enumerate(lines):
            self.text(x+w/2, y+h/2 + (i-(len(lines)-1)/2)*gap, line,
                      size=size, color=color, bold=i == 0)
            font = ImageFont.truetype(str(FONT), size)
            if font.getlength(line) > w-16:
                self.errors.append(f'text exceeds box {id}: {line}')
        return id

    def circle(self, id, x, y, size=38, label='+', color=ORANGE_INK):
        return self.box(id, x-size/2, y-size/2, size, size, [label],
                        '#ffffff', 27, color, size/2, color, kind='circle')

    def port(self, id, side='right'):
        b = self.boxes[id]
        return {'left': (b['x'], b['y']+b['h']/2),
                'right': (b['x']+b['w'], b['y']+b['h']/2),
                'top': (b['x']+b['w']/2, b['y']),
                'bottom': (b['x']+b['w']/2, b['y']+b['h'])}[side]

    def path(self, pts, color=LINE, width=2.2, arrow=True, dashed=False):
        self.items.append(dict(kind='path', pts=pts, color=color,
                               width=width, arrow=arrow, dashed=dashed))

    def connect(self, src, dst, a='right', b='left', via=(), **kw):
        self.path([self.port(src, a), *via, self.port(dst, b)], **kw)

    def panel(self, id, y, h, title, subtitle=''):
        self.box(id, 35, y, self.w-70, h, (), '#ffffff', stroke=BORDER,
                 radius=14, kind='panel')
        self.text(65, y+35, title, 28, anchor='start', bold=True)
        if subtitle:
            self.text(self.w-65, y+35, subtitle, 20, MUTED, 'end')

    def export(self, stem, scale=1.5):
        if self.errors:
            raise ValueError('\n'.join(self.errors))
        # Paint panel backgrounds first, then connections, then nodes and labels.
        ordered = ([o for o in self.items if o['kind'] == 'panel'] +
                   [o for o in self.items if o['kind'] == 'path'] +
                   [o for o in self.items if o['kind'] not in ('panel', 'path')])
        im = Image.new('RGB', (round(self.w*scale), round(self.h*scale)), 'white')
        draw = ImageDraw.Draw(im)
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">',
               f'<title>{html.escape(self.title)}</title>',
               '<rect width="100%" height="100%" fill="white"/>']
        editor = dict(nodes=[], edges=[], notes=[], settings=dict(corner=10,
                      gridSize=5, snap=True, lastBulge=100, lastRoute='straight',
                      dragDelay=180, routeMemory={}), profileGroups={},
                      palette='past', view=dict(x=0, y=0, w=self.w, h=self.h))
        model = ET.Element('mxGraphModel', adaptiveColors='none',
                           page='1', pageScale='1', pageWidth=str(self.w),
                           pageHeight=str(self.h), background='#ffffff')
        root = ET.SubElement(model, 'root')
        ET.SubElement(root, 'mxCell', id='0')
        ET.SubElement(root, 'mxCell', id='1', parent='0')
        def geom(cell, **attrs):
            ET.SubElement(cell, 'mxGeometry', {**{k: str(v) for k,v in attrs.items()}, 'as':'geometry'})
        for idx, o in enumerate(ordered):
            kind = o['kind']
            if kind in ('box', 'panel', 'circle'):
                x,y,w,h = (o[k] for k in ('x','y','w','h'))
                fill, stroke = o['fill'], o['stroke']
                xy = [round(v*scale) for v in (x,y,x+w,y+h)]
                if kind == 'circle':
                    draw.ellipse(xy, fill=fill, outline=stroke, width=round(1.6*scale))
                    svg.append(f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w/2}" ry="{h/2}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
                else:
                    draw.rounded_rectangle(xy, radius=round(o['radius']*scale), fill=fill, outline=stroke, width=round(1.3*scale))
                    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{o["radius"]}" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
                editor['nodes'].append(dict(id=o['id'], type='residual' if kind=='circle' else 'rect', x=x,y=y,w=w,h=h,label='', fill=fill, textColor=INK, orientation='horizontal', radius=o['radius']))
                style = f'html=1;whiteSpace=wrap;rounded=1;arcSize=10;fillColor={fill};strokeColor={stroke};strokeWidth=1.3;'
                if kind == 'circle': style += 'shape=ellipse;'
                cell = ET.SubElement(root,'mxCell',id=o['id'],value='',style=style,vertex='1',parent='1')
                geom(cell,x=x,y=y,width=w,height=h)
            elif kind == 'text':
                x,y,t,size = (o[k] for k in ('x','y','text','size'))
                font = ImageFont.truetype(str(FONT), round(size*scale))
                anchor = {'middle':'mm','start':'lm','end':'rm'}[o['anchor']]
                draw.text((x*scale,y*scale), t, font=font, fill=o['color'], anchor=anchor)
                svg.append(f'<text x="{x}" y="{y}" text-anchor="{o["anchor"]}" dominant-baseline="central" font-family="Arial Unicode MS, PingFang SC, Microsoft YaHei, sans-serif" font-size="{size}" fill="{o["color"]}">{html.escape(t)}</text>')
                noteid=f'text_{idx}'
                # The editor supports centered notes; translate other anchors.
                tw = ImageFont.truetype(str(FONT), size).getlength(t)
                nx = x + (tw/2 if o['anchor']=='start' else -tw/2 if o['anchor']=='end' else 0)
                editor['notes'].append(dict(id=noteid,x=nx,y=y,text=t,fontSize=size,color=o['color'],parentId=None,anchor='free',dx=0,dy=0))
                cell = ET.SubElement(root,'mxCell',id=noteid,value=html.escape(t),
                                     style=f'text;html=1;whiteSpace=wrap;align={"left" if o["anchor"]=="start" else "right" if o["anchor"]=="end" else "center"};verticalAlign=middle;fontFamily=Arial Unicode MS;fontSize={size};fontColor={o["color"]};spacing=0;',vertex='1',parent='1')
                tx = x if o['anchor']=='start' else x-tw if o['anchor']=='end' else x-tw/2
                geom(cell,x=tx-2,y=y-size*0.8,width=tw+4,height=size*1.6)
            elif kind == 'path':
                pts, color = o['pts'], o['color']
                points = [(x*scale,y*scale) for x,y in pts]
                dash = ' stroke-dasharray="7 5"' if o['dashed'] else ''
                svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in pts)+f'" fill="none" stroke="{color}" stroke-width="{o["width"]}" stroke-linejoin="round"{dash}/>')
                for p,q in zip(points,points[1:]):
                    if o['dashed']:
                        dx,dy=q[0]-p[0],q[1]-p[1]; length=math.hypot(dx,dy)
                        for offset in range(0,round(length),round(12*scale)):
                            t0=offset/max(length,1);t1=min((offset+7*scale)/max(length,1),1)
                            draw.line([(p[0]+dx*t0,p[1]+dy*t0),(p[0]+dx*t1,p[1]+dy*t1)],fill=color,width=round(o['width']*scale))
                    else:
                        draw.line([p,q],fill=color,width=round(o['width']*scale))
                if o['arrow']:
                    a,b=pts[-2:]; dx,dy=b[0]-a[0],b[1]-a[1]; length=math.hypot(dx,dy)
                    if length:
                        ux,uy=dx/length,dy/length
                        tip=[b,(b[0]-9*ux+4*uy,b[1]-9*uy-4*ux),(b[0]-9*ux-4*uy,b[1]-9*uy+4*ux)]
                        draw.polygon([(x*scale,y*scale) for x,y in tip],fill=color)
                        svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in tip)+f'" fill="{color}"/>')
                for j,(p,q) in enumerate(zip(pts,pts[1:])):
                    eid=f'edge_{idx}_{j}'
                    arrow=o['arrow'] and j==len(pts)-2
                    editor['edges'].append(dict(id=eid, **{'from':dict(x=p[0],y=p[1]), 'to':dict(x=q[0],y=q[1])}, route='straight',arrow=arrow,corner=0,width=o['width'],color=color))
                cell=ET.SubElement(root,'mxCell',id=f'edge_{idx}',edge='1',parent='1',style=f'html=1;rounded=0;strokeColor={color};strokeWidth={o["width"]};endArrow={"block" if o["arrow"] else "none"};endFill=1;'+('dashed=1;' if o['dashed'] else ''))
                g=ET.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
                ET.SubElement(g,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),attrib={'as':'sourcePoint'})
                ET.SubElement(g,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),attrib={'as':'targetPoint'})
                if len(pts)>2:
                    arr=ET.SubElement(g,'Array',attrib={'as':'points'})
                    for x,y in pts[1:-1]: ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
        svg.append('</svg>')
        (ROOT/f'{stem}.svg').write_text('\n'.join(svg),encoding='utf-8')
        im.save(ROOT/f'{stem}.png',optimize=True)
        (ROOT/f'{stem}.json').write_text(json.dumps(editor,ensure_ascii=False,indent=2),encoding='utf-8')
        mxfile=ET.Element('mxfile',host='app.diagrams.net')
        ET.SubElement(mxfile,'diagram',name=self.title,id=stem).append(model)
        ET.indent(mxfile)
        ET.ElementTree(mxfile).write(ROOT/f'{stem}.drawio',encoding='utf-8',xml_declaration=True)
        print(f'{stem}: {self.w}×{self.h}; {len(editor["nodes"])} shapes; {len(editor["edges"])} connector segments')


def overview():
    s=Scene(1440,860,'保留电极通道的双域 EEG-Conformer · 方案总览')
    s.text(45,45,'保留电极通道的双域 EEG-Conformer',34,anchor='start',bold=True)
    s.text(45,89,'每路单一深度 10 层：前 5 层逐电极处理 → 融合 21→1 → 后 5 层',24,MUTED,'start')
    s.box('input',35,325,170,110,['EEG 窗口','[B,1,21,1920]'],GREY,22)
    for p,y,name,n,flat,fill in [('time',220,'Time 时域',122,4880,BLUE),('fft',490,'FFT 频域',58,2320,PURPLE)]:
        s.box(p+'prep',255,y,195,100,[name,'标准化' if p=='time' else 'rFFT → 对数功率'],fill,23)
        s.box(p+'patch',500,y,220,100,['PatchEmbedding','保留 21 个电极',f'[B,21,{n},40]'],fill,22)
        s.box(p+'enc',780,y-10,355,120,['10 层 Encoder','前 5 层 → 21→1 → 后 5 层',f'[B,21,{n},40] → [B,{n},40]'],GREEN,21)
        s.box(p+'skip',780,y-88,355,48,['入口卷积 21→21 → 前 5 层各末尾'],ORANGE,19,color=ORANGE_INK)
        s.box(p+'flat',1190,y,200,100,['Flatten',f'[B,{flat}]'],TEAL,24)
        s.connect('input',p+'prep',via=[(228,380),(228,y+50)])
        s.connect(p+'prep',p+'patch')
        s.connect(p+'patch',p+'enc')
        s.connect(p+'enc',p+'flat')
        s.connect(p+'patch',p+'skip',a='top',b='left',via=[(610,y-64)],color=ORANGE_INK)
        s.connect(p+'skip',p+'enc',a='bottom',b='top',color=ORANGE_INK)
    s.box('cat',780,700,250,85,['Time / FFT 拼接','[B,7200]'],TEAL,24)
    s.box('head',1090,700,300,85,['单一分类头 → 3 类','e_1 / e_2 / e_3'],BLUE,24)
    s.connect('timeflat','cat',a='right',b='top',via=[(1410,270),(1410,650),(905,650)])
    s.connect('fftflat','cat',a='right',b='bottom',via=[(1425,540),(1425,807),(905,807)])
    s.connect('cat','head')
    s.text(45,675,'维度说明',23,anchor='start',bold=True)
    s.text(45,715,'21：电极通道',23,MUTED,'start')
    s.text(45,753,'122 / 58：两路 token 数',23,MUTED,'start')
    s.text(45,791,'40：学习到的特征维',23,MUTED,'start')
    s.text(45,837,'依据录音最终方案；跳接相加、跨电极共享 Encoder 参数及末端双域拼接为绘图假设。',19,MUTED,'start')
    return s


def detailed():
    s=Scene(2400,2050,'保留电极通道的 Time + FFT 双域 EEG-Conformer · 10层方案')
    s.text(65,48,'保留电极通道的 Time + FFT 双域 EEG-Conformer',39,anchor='start',bold=True)
    s.text(65,94,'录音最终方案重绘 · 每路 depth=10 · 前 5 层保留 C=21，之后融合 · 方案图，尚未修改训练模型',25,MUTED,'start')
    s.panel('overview_panel',135,490,'① 总体结构','示例输入沿用原图；Time / FFT 参数各自独立')
    s.box('raw',65,337,190,108,['EEG window','[B,1,21,1920]'],GREY,23)
    for p,y,n,flat,fill in [('time',235,122,4880,BLUE),('fft',455,58,2320,PURPLE)]:
        s.box(p+'pre',320,y,220,100,['Time 标准化' if p=='time' else 'rFFT → log1p 功率','[B,1,21,1920]' if p=='time' else '标准化 [B,1,21,961]'],fill,21)
        s.box(p+'stem',600,y,235,100,['保留通道的 Stem',f'[B,21,{n},40]'],fill,23)
        s.box(p+'first',895,y,240,100,['Block #1–#5','逐电极序列建模','入口卷积逐层跳接'],GREEN,23)
        s.box(p+'merge',1195,y,185,100,['跨电极融合','C: 21 → 1'],ORANGE,23,color=ORANGE_INK)
        s.box(p+'last',1440,y,235,100,['Block #6–#10',f'[B,{n},40]'],GREEN,23)
        s.box(p+'flatten',1735,y,180,100,['Flatten',f'[B,{flat}]'],TEAL,23)
        s.connect('raw',p+'pre',via=[(285,391),(285,y+50)])
        for a,b in [('pre','stem'),('stem','first'),('first','merge'),('merge','last'),('last','flatten')]:s.connect(p+a,p+b)
    s.box('fusion',1975,345,160,100,['特征拼接','[B,7200]'],TEAL,23)
    s.box('classifier',2195,345,140,100,['单一 MLP','3 类输出'],BLUE,22)
    s.connect('timeflatten','fusion',via=[(1945,285),(1945,395)])
    s.connect('fftflatten','fusion',via=[(1930,505),(1930,395)])
    s.connect('fusion','classifier')
    s.text(1200,592,'每域由 11 / 10 / 8 三条深度链改为一条 10 层链；跨深度 QKV、深度权重与三组深度分类头随之取消。',22,MUTED)

    s.panel('stem_panel',650,340,'② PatchEmbedding：保留电极维 C，不在入口把 21 压成 1','C=电极；D=特征；L=原始长度；N=token')
    stem=[('si',65,190,['输入','[B,1,21,L]'],GREY),
          ('sc',315,270,['时间 / 频率轴卷积','Conv2d 1→40, k=(1,25)','[B,40,21,L−24]'],BLUE),
          ('sn',645,215,['BatchNorm + ELU','C 仍为 21'],GREEN),
          ('sp',920,280,['AvgPool + Dropout','k=(1,75), stride=(1,15)','[B,40,21,N]'],ORANGE),
          ('sj',1260,265,['1×1 特征投影','40 → D=40','[B,40,21,N]'],BLUE),
          ('st',1585,285,['Permute，保留 C','[B,D,C,N] → [B,C,N,D]','此处不做 squeeze'],TEAL),
          ('so',1930,400,['输出给 Encoder','Time [B,21,122,40]','FFT [B,21,58,40]'],TEAL)]
    for id,x,w,lines,fill in stem:s.box(id,x,745,w,123,lines,fill,22)
    for a,b in zip(stem,stem[1:]):s.connect(a[0],b[0])
    s.text(65,920,'移走原 Stem 的跨电极空间卷积 Conv2d(40→40, k=(21,1))；仅删除 squeeze 无法恢复已被融合的电极维。',23,ORANGE_INK,'start')
    s.text(65,958,'N = floor((L−24−75)/15)+1；L_Time=1920 → N=122；L_FFT=961 → N=58。40 来自卷积滤波器 / 嵌入维。',22,MUTED,'start')

    s.panel('encoder_panel',1015,600,'③ 单域 Encoder 展开：Time 和 FFT 各运行一套','下图 N=122（Time）或 58（FFT）；橙色线为新增跨电极信息')
    s.text(65,1095,'输入 X₀ = [B,21,N,40]',26,anchor='start',bold=True)
    s.box('skip_conv',410,1065,405,90,['入口跨电极卷积 21→21','S = Conv_C(X₀)，N 与 D 不变'],ORANGE,23,color=ORANGE_INK)
    s.path([(330,1095),(370,1095),(370,1110),(410,1110)],color=ORANGE_INK)
    s.path([(612.5,1155),(612.5,1190),(2290,1190)],color=ORANGE_INK,arrow=False)
    s.text(1630,1093,'同一 S 分别送到前 5 个 Block 的末尾',24,ORANGE_INK)
    xs=[360,750,1140,1530,1920]
    for i,x in enumerate(xs,1):
        s.box(f'early_{i}',x,1240,230,100,[f'Block #{i}','逐电极 [N,40]'],GREEN,23)
        s.circle(f'add_{i}',x+285,1290,42)
        s.connect(f'early_{i}',f'add_{i}')
        s.path([(x+285,1190),(x+285,1269)],color=ORANGE_INK)
        if i>1:s.connect(f'add_{i-1}',f'early_{i}')
    s.path([(115,1118),(115,1290),(360,1290)])
    s.text(1200,1390,'各层输出均为 [B,21,N,40]；H_i = Block_i(H_(i−1)) + S，i=1…5（相加为绘图假设）',23,MUTED)
    s.box('mid_fuse',1920,1460,325,100,['第 5 层后：跨电极融合','21→1，再 squeeze 电极维','[B,21,N,40] → [B,N,40]'],ORANGE,22,color=ORANGE_INK)
    s.connect('add_5','mid_fuse',a='bottom',b='top',via=[(2205,1420),(2082.5,1420)])
    s.box('late_6',1530,1460,270,100,['Block #6','融合后继续建模'],GREEN,23)
    s.box('late_7_9',1140,1460,270,100,['Block #7–#9','[B,N,40]'],GREEN,23)
    s.box('late_10',750,1460,270,100,['Block #10','[B,N,40]'],GREEN,23)
    s.box('enc_out',360,1460,270,100,['Flatten 输出','Time 4880 / FFT 2320'],TEAL,22)
    s.connect('mid_fuse','late_6',a='left',b='right')
    s.connect('late_6','late_7_9',a='left',b='right')
    s.connect('late_7_9','late_10',a='left',b='right')
    s.connect('late_10','enc_out',a='left',b='right')

    s.panel('details_panel',1640,365,'④ Block 与分类头：沿用基线结构，默认超参数先回归','depth=10 为导师指定；仓库默认 depth=6，因此不能全部直接用默认值')
    s.box('block_detail',65,1730,740,160,['单个 TransformerEncoderBlock','x₁ = x + Dropout(MHA(LayerNorm(x)))','out = x₁ + Dropout(FFN(LayerNorm(x₁)))','D=40；heads=5；FFN=40→160→40；dropout=0.5'],GREEN,24)
    s.box('head_detail',845,1730,670,160,['末端分类（双域拼接按原图保留）','4880 + 2320 → 7200 → 256 → 32 → 3','ELU；分类头 dropout=0.5 / 0.3','单一三分类交叉熵；不再做三组深度损失加权'],TEAL,23)
    s.box('assumptions',1555,1730,775,160,['绘图假设 / 待实现确认','各电极共享 Encoder 参数，可展平为 [B×21,N,40]','跳接暂按相加；若用拼接，需要补投影与维度定义','21→21 与 21→1 的卷积核 / 分组方式未在录音定死'],ORANGE,23,color=ORANGE_INK)
    s.text(65,1940,'基线出处：train_activity_loso.py。优化器默认 Adam，lr=0.0002，betas=(0.5,0.999)；旧 0.85 Encoder dropout 不沿用。',22,MUTED,'start')
    s.text(65,1980,'图中省略了禁用的 Input QKV / FFT-global。类别权重、重采样与训练日程不由本图另行确定；此图不表示模型已实现或验证有效。',21,MUTED,'start')
    s.text(65,2030,'设计依据：录音 16:38–16:49、19:19–24:38、25:17–27:51、28:32–29:20；结合原结构图和仓库模型定义核对。',20,MUTED,'start')
    return s


if __name__ == '__main__':
    overview().export('框架总览')
    detailed().export('详细框架')
