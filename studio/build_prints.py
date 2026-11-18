"""Build the README plates from make_viz_data.py's browser snapshot."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'viz_data.js').read_text().split('=', 1)[1].strip(' ;\n'))
probs = data['ref_p']


def plate(name, title, caption, body):
    (ROOT / 'assets' / (name + '.svg')).write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 388" role="img"><title>{title}</title><desc>{caption}</desc>
<rect width="960" height="388" rx="12" fill="#101b22"/><g font-family="monospace" fill="#e9e1cd"><text x="38" y="40" font-size="16" letter-spacing="3">FIELD NOTES  /  QAOA LAB</text><text x="38" y="90" font-family="serif" font-size="44">{title}</text>{body}<path d="M38 335H922" stroke="#536264"/><text x="38" y="364" font-size="17">{caption}</text></g></svg>''')


# A skyline of order statistics. Height has a shared, linear probability scale.
bars = []
for i, p in enumerate(sorted(probs)):
    h = p * 209
    color = '#ffad74' if p > .5 else '#83d9c3'
    bars.append(f'<path d="M{40+i*1.75} 312v-{h:.2f}" stroke="{color}" stroke-width="1.6"/>')
plate('atlas', 'Атлас пятисот задач', '01   /   Слева направо: от меньшей P к большей.  Охра — выше 50%.', ''.join(bars) + '<path d="M38 207.5H918" stroke="#eee" stroke-dasharray="3 7"/><text x="40" y="198">50%</text>')

# Signed angle traces on the same radial scale, ordered by circuit layer.
body = []
ids = sorted(range(len(probs)), key=probs.__getitem__)
for col, idx in enumerate([ids[0], ids[len(ids)//2], ids[-1]]):
    x, y = 170+col*310, 216
    for r in [30, 55, 80]:
        body.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="#39474d"/>')
    for off,color in [(0,'#83d9c3'),(5,'#ffad74')]:
        points=[]
        for j,a in enumerate(data['angles'][idx][off:off+5]):
            t=math.tau*j/5-math.pi/2
            r=55+7*a
            points.append(f'{x+r*math.cos(t):.2f},{y+r*math.sin(t):.2f}')
        body.append(f'<polygon points="{" ".join(points)}" fill="{color}" fill-opacity=".09" stroke="{color}" stroke-width="3"/>')
    body.append(f'<text x="{x}" y="319" text-anchor="middle">№ {idx}  /  P {100*probs[idx]:.1f}%</text>')
plate('orbits', 'Паспорта углов', '02   /   Пять осей — пять слоёв.  Мята: γ.  Охра: β.  Среднее кольцо: 0 рад.', ''.join(body))
