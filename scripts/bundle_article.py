"""Bundle docs/article/index.html + fig/*.png into one self-contained HTML (images as data URIs).

  python scripts/bundle_article.py   ->  docs/article/ICH_VisionHOPE_article.html
"""
import base64
import re
from pathlib import Path

d = Path(__file__).resolve().parents[1] / "docs" / "article"
html = (d / "index.html").read_text(encoding="utf-8")


def inline(m):
    f = d / m.group(1)
    return 'src="data:image/png;base64,' + base64.b64encode(f.read_bytes()).decode() + '"'


out = re.sub(r'src="(fig/[^"]+\.png)"', inline, html)
if not out.lstrip().lower().startswith("<!doctype"):
    out = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           '<style>body{margin:0}img{max-width:100%}</style>\n</head>\n<body>\n' + out + "\n</body>\n</html>\n")
dst = d / "ICH_VisionHOPE_article.html"
dst.write_text(out, encoding="utf-8")
print(f"wrote {dst} ({dst.stat().st_size / 2**20:.1f} MB)")
