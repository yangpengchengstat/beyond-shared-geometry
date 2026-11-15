#!/usr/bin/env python3
"""Plot saved endpoints only. No activations, fitting, resampling, or new risks."""
from pathlib import Path
import argparse
import csv
import hashlib
import json


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=here.parents[1] / 'figures' / 'same_metric_risks_v061.pdf')
    args = parser.parse_args()
    bindings = json.loads((here / 'INPUT_MAPPING.json').read_text())
    inputs = {}
    for name, record in bindings['inputs'].items():
        p = here / name
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != record['sha256']:
            raise ValueError('Saved input hash mismatch: ' + name)
        with p.open(newline='') as f:
            inputs[name] = list(csv.DictReader(f))
    risks = inputs['cell_risk_reference.csv']
    orth_rows = [row for row in inputs['orthogonal_cell_summary.csv'] if row['dimension'] == '16']
    orth = {row['cell']: row for row in orth_rows}
    if len(risks) != 48 or len(orth_rows) != 48 or len(orth) != 48:
        raise ValueError('Expected exactly 48 primary conditions in each saved source')
    if {r['cell'] for r in risks} != set(orth):
        raise ValueError('Condition identities do not match')

    import math
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    models = [
        ('meta-llama/Llama-3.2-1B', ('Llama', '3.2-1B')),
        ('meta-llama/Llama-3.2-3B', ('Llama', '3.2-3B')),
        ('meta-llama/Llama-3.1-8B', ('Llama', '3.1-8B')),
        ('Qwen/Qwen2.5-1.5B', ('Qwen2.5', '1.5B')),
        ('Qwen/Qwen2.5-3B', ('Qwen2.5', '3B')),
        ('Qwen/Qwen2.5-7B', ('Qwen2.5', '7B')),
    ]
    plotted = []
    for domain in ('C7', 'D4'):
        for model, _ in models:
            cells = sorted((r for r in risks if r['domain'] == domain and r['model_id'] == model), key=lambda r: int(r['layer']))
            if len(cells) != 4:
                raise ValueError('Expected four readouts per model/domain')
            for row in cells:
                o = orth[row['cell']]
                if o['status'] != 'complete' or o['domain'] != domain or o['model_id'] != row['model_id'] or o['layer'] != row['layer']:
                    raise ValueError('Mismatched saved condition')
                plotted.append({'cell': row['cell'], 'domain': domain, 'model_id': row['model_id'],
                                'layer': row['layer'], 'dimension': '16', 'eta': row['eta'],
                                'L_orth': o['L_orth'], 'L_lower': row['a'], 'L_upper': row['u'],
                                'L_identity': row['identity_risk']})
    risk_keys = ('eta', 'L_orth', 'L_lower', 'L_upper', 'L_identity')
    # Axis limits only: every plotted endpoint is an unchanged CSV value.
    values = [float(row[k]) for row in plotted for k in risk_keys]
    if min(values) <= 0:
        raise ValueError('Log axis requires positive saved risks')
    emin, emax = math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))
    width, height, left, right, ph = 540, 405, 48, 534, 120
    args.output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(args.output), pagesize=(width, height), invariant=1)
    c.setTitle('Existing same-metric risks, all 48 primary conditions')
    c.setAuthor('Anonymous authors')
    colors = {'eta':'#2466A5', 'L_orth':'#B76A0C', 'bound':'#247B51', 'L_identity':'#343434'}
    def text(x, y, label, size=8, align='center', bold=False, color='#263238'):
        c.setFillColor(HexColor(color)); c.setFont('Helvetica-Bold' if bold else 'Helvetica', size)
        getattr(c, {'center':'drawCentredString', 'left':'drawString', 'right':'drawRightString'}[align])(x, y, label)
    def line(x1,y1,x2,y2,color='#D7DCE0',lw=.5):
        c.setStrokeColor(HexColor(color));c.setLineWidth(lw);c.line(x1,y1,x2,y2)
    def marker(x,y,kind):
        color=colors[kind];c.setStrokeColor(HexColor(color));c.setFillColor(HexColor(color));c.setLineWidth(.9)
        if kind=='eta':c.circle(x,y,1.8,stroke=0,fill=1)
        elif kind=='L_orth':
            path=c.beginPath();path.moveTo(x,y+2.5);path.lineTo(x-2.3,y-1.6);path.lineTo(x+2.3,y-1.6);path.close();c.drawPath(path,stroke=0,fill=1)
        else:
            c.line(x-2,y-2,x+2,y+2);c.line(x-2,y+2,x+2,y-2)
    for x,y,kind,label in [(65,391,'eta','Linear optimum'),(278,391,'L_orth','Independent orthogonal optimum'),(278,377,'L_identity','Identity-family risk')]:
        marker(x,y+2,kind);text(x+9,y,label,size=8.7,align='left')
    line(65,375,65,384,colors['bound'],1.5);line(62,375,68,375,colors['bound'],1);line(62,384,68,384,colors['bound'],1)
    text(74,377,'Exact-action class-risk bounds',8.7,align='left')
    for domain, y0, title in [('C7',228,'Weekdays (C7)'),('D4',56,'Corners (D4)')]:
        rows=[r for r in plotted if r['domain']==domain]
        dx=(right-left)/24
        X=lambda j:left+dx*(j+.5)
        Y=lambda val:y0+ph*(math.log10(val)-emin)/(emax-emin)
        for k,(_, labels) in enumerate(models):
            if k%2==0:
                c.setFillColor(HexColor('#F0F3F5'));c.rect(left+4*k*dx,y0,4*dx,ph,stroke=0,fill=1)
            for n,label in enumerate(labels):text(left+(4*k+2)*dx,y0-25-9*n,label,8)
        for power in range(emin,emax+1):
            y=Y(10**power);line(left,y,right,y);text(left-6,y-3,'1' if power==0 else f'10^{power}',8,align='right')
        line(left,y0,left,y0+ph,'#59636D',.7);line(left,y0,right,y0,'#59636D',.7)
        for k in range(1,6):line(left+4*k*dx,y0,left+4*k*dx,y0+ph,'#CED5DA',.6)
        text(left,y0+ph+9,title,10,align='left',bold=True)
        c.saveState();c.translate(11,y0+ph/2);c.rotate(90);text(0,0,'Whitened risk (log scale)',8.5);c.restoreState()
        for j,row in enumerate(rows):
            x=X(j)
            marker(x-4.0,Y(float(row['eta'])),'eta');marker(x-.9,Y(float(row['L_orth'])),'L_orth')
            lo,hi=Y(float(row['L_lower'])),Y(float(row['L_upper']))
            line(x+2.3,lo,x+2.3,hi,colors['bound'],1.5)
            line(x+.6,lo,x+4,lo,colors['bound'],1);line(x+.6,hi,x+4,hi,colors['bound'],1)
            marker(x+6.2,Y(float(row['L_identity'])),'L_identity')
            text(x,y0-11,('.2','.4','.6','.8')[j%4],7)
    text((left+right)/2,7,'Within each model: normalized depth slots 0.2, 0.4, 0.6, 0.8',8)
    c.showPage();c.save()
    points = here / 'plotted_endpoints.csv'
    with points.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(plotted[0])); writer.writeheader(); writer.writerows(plotted)
    record = {'operation': 'join_and_plot_saved_endpoints_only', 'dimension': 16, 'conditions': 48,
              'no_new_risks_or_statistics': True, 'no_experiments': True,
              'source_bindings': 'INPUT_MAPPING.json', 'output': 'paper/figures/' + args.output.name,
              'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'plotted_endpoints_sha256': hashlib.sha256(points.read_bytes()).hexdigest(),
              'no_joined_witness_transfer_metric': True, 'renderer': 'ReportLab vector graphics',
              'display': 'logarithmic y axis; horizontal offsets distinguish endpoints, not new conditions'}
    (here / 'PRESENTATION_PROVENANCE.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
