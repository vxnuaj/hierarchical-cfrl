"""Convert notes/theory.md (sections 0-10) into the LaTeX theory appendix paper/theory_appendix.tex.

Numbering: every theorem-like label and section reference of the notes gets a "T" prefix (Theorem 3.1 -> Theorem T3.1,
section 4 -> section T4) so it cannot be confused with the paper's own numbering.
usage: python scripts/theory_md2tex.py
"""
import re

SRC, DST = "notes/theory.md", "paper/theory_appendix.tex"
CITE = {
    "Ali, Kolter & Tibshirani 2019": "ali2019earlystop", "Ali, Kolter & Tibshirani 2019": "ali2019earlystop",
    "Amari et al. 2021": "amari2021precond", "Balles & Hennig 2018": "balles2018dissecting",
    "Bernstein & Newhouse 2024": "bernstein2024oldnorm", "Jordan et al. 2024": "jordan2024muon",
    "Bernstein & Newhouse, *Modular duality*, 2024": "bernstein2024modular",
    "Eisenstein et al. 2023": "eisenstein2023helping", "Fan, Schmidt & Thrampoulidis 2025": "fan2025spectral",
    "Gao et al. 2023": "gao2023scaling", "Ilyas et al. 2019": "ilyas2019features",
    "Panickssery et al. 2024": "panickssery2024llm", "Wadia et al. 2021": "wadia2021whitening",
    "LeCun, Kanter & Solla 1991": "lecun1991eigenvalues", "Advani & Saxe 2020": "advani2020highdim",
}
UNI = {"–": "--", "—": ",", "−": "$-$", "→": "$\\to$", "≈": "$\\approx$", "′": "$'$", "×": "$\\times$",
       "…": "\\dots{}", "§": "\\S{}", "≤": "$\\le$", "≪": "$\\ll$"}


def tprefix(s):
    s = re.sub(r"\b(Theorems?|Lemmas?|Propositions?|Corollar(?:y|ies)|Thm|Prop|Cor|Lemma)(\s+|~)(\d)", r"\1\2T\3", s)
    s = re.sub(r"\b(and|or|,|/|–|-)\s*(\d\.\d)(?=[^\d])", lambda m: m.group(0), s)
    s = re.sub(r"\b([Ss]ections?)\s+(\d+)", r"\1 T\2", s)
    s = re.sub(r"§\s?(\d+)", r"§T\1", s)
    # lists like "Theorem 3.1 and 4.1", "Lemmas 2.1–2.3", "Props 5.1–5.4": prefix the later numbers too
    s = re.sub(r"(T\d\.\d+′?)(\s*(?:–|-|,|and|or|/)\s*)(\d\.\d)", r"\1\2T\3", s)
    s = re.sub(r"(T\d\.\d+′?)(\s*(?:–|-|,|and|or|/)\s*)(\d\.\d)", r"\1\2T\3", s)
    return s


def cites(s):
    def rep(m):
        parts = [p.strip() for p in m.group(1).split(";")]
        keys = [CITE.get(p) for p in parts]
        if all(keys):
            return "\\citep{" + ",".join(keys) + "}"
        return m.group(0)
    return re.sub(r"\[([A-Z][^\[\]]*?\d{4}[a-z]?)\]", rep, s)


def text_escape(t):
    out = []
    for piece in re.split(r"(`[^`]*`)", t):
        if piece.startswith("`") and piece.endswith("`") and len(piece) > 1:
            body = piece[1:-1].replace("\\", "/").replace("_", "\\_").replace("&", "\\&").replace("{", "\\{").replace("}", "\\}")
            out.append("\\texttt{" + body + "}")
            continue
        p = piece.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#").replace("_", "\\_")
        p = re.sub(r"\*\*\[([^\]]+)\]\*\*", r"\\textsc{[\1]}", p)
        p = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", p)
        p = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", p)
        p = p.replace(" — ", ", ").replace("—", ", ")
        for a, b in UNI.items():
            p = p.replace(a, b)
        out.append(p)
    return "".join(out)


def convert_inline(s):
    """Escape a run of text while leaving $...$ and \\[...\\] math untouched."""
    s = tprefix(s)
    s = cites(s)
    pieces = re.split(r"(\\\[.*?\\\]|\$[^$]+\$)", s, flags=re.S)
    res = []
    for i, p in enumerate(pieces):
        if i % 2 == 1:
            p = p.replace("−", "-").replace("′", "'").replace("…", "\\dots").replace("≈", "\\approx").replace("×", "\\times").replace("→", "\\to").replace("≤", "\\le")
            res.append(p)
        else:
            res.append(text_escape(p))
    return "".join(res)


def table(lines):
    rows = [[c.strip().replace("@@BAR@@", "\\|") for c in l.replace("\\|", "@@BAR@@").strip().strip("|").split("|")] for l in lines if not re.match(r"^\|\s*-", l)]
    n = len(rows[0])
    widths = {3: ["4.4cm", "7.8cm", "3.6cm"], 2: ["5cm", "10.5cm"]}.get(n, [f"{(15.2 - 0.22 * n) / n:.2f}cm"] * n)
    spec = "".join(f">{{\\raggedright\\arraybackslash}}p{{{w}}}" for w in widths)
    body = ["\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
            " & ".join("\\textbf{" + convert_inline(c) + "}" for c in rows[0]) + " \\\\", "\\midrule"]
    for r in rows[1:]:
        body.append(" & ".join(convert_inline(c) for c in r) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    return "\n".join(body)


def main():
    md = open(SRC).read()
    start = md.index("## 0. Setting and notation")
    end = md.index("## References")
    md = md[start:end]
    md = re.sub(r"\$\$(.+?)\$\$", lambda m: "\\[" + m.group(1).strip() + "\\]", md, flags=re.S)
    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        l = lines[i]
        if l.strip() == "---" or not l.strip():
            out.append("")
            i += 1
            continue
        m = re.match(r"^## (\d+)\. (.*)", l)
        if m:
            out.append(f"\\subsection*{{T{m.group(1)}. {convert_inline(m.group(2))}}}")
            out.append(f"\\label{{app:theory-{m.group(1)}}}")
            i += 1
            continue
        if l.startswith("|"):
            blk = []
            while i < len(lines) and lines[i].startswith("|"):
                blk.append(lines[i]); i += 1
            out.append(table(blk))
            continue
        m = re.match(r"^(\s*)(- |\d+\. )(.*)", l)
        if m and not m.group(1):
            env = "itemize" if m.group(2) == "- " else "enumerate"
            items, cur = [], None
            while i < len(lines):
                mm = re.match(r"^(- |\d+\. )(.*)", lines[i])
                if mm:
                    if cur is not None: items.append(cur)
                    cur = mm.group(2); i += 1
                elif lines[i].startswith("  ") and lines[i].strip() and cur is not None:
                    cur += " " + lines[i].strip(); i += 1
                else:
                    break
            items.append(cur)
            out.append(f"\\begin{{{env}}}\\setlength{{\\itemsep}}{{1pt}}")
            out += ["\\item " + convert_inline(it) for it in items]
            out.append(f"\\end{{{env}}}")
            continue
        blk = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("|") and not re.match(r"^(## |- |\d+\. |---)", lines[i]):
            blk.append(lines[i]); i += 1
        para = "\n".join(blk)
        para = re.sub(r"^\*\*((?:Theorem|Lemma|Proposition|Corollary|Definition)[^*]*?)\*\*", r"@@HEAD@@\1@@", para)
        conv = convert_inline(para)
        conv = re.sub(r"@@HEAD@@(.*?)@@", lambda m: "\\paragraph{" + m.group(1).rstrip(".") + ".}", conv, flags=re.S)
        out.append(conv)
    tex = "\n".join(out)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    open(DST, "w").write(tex + "\n")
    print("wrote", DST, len(tex.split()), "words")


if __name__ == "__main__":
    main()
