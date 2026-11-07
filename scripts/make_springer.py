"""Convert paper/ids.tex into the Springer Nature journal template (sn-jnl, numbered references).

The Springer template wants one .tex file, so the generated paper/springer/ids_sn.tex inlines the tables of
paper/tables. The class and bibliography style come from the Springer template package (copied once to
paper/springer/: sn-jnl.cls and sn-mathphys-num.bst). Run after the table scripts; then build with
    cd paper/springer && pdflatex ids_sn && bibtex ids_sn && pdflatex ids_sn && pdflatex ids_sn
Usage: python scripts/make_springer.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

PAPER = Path("paper")
OUT = PAPER / "springer"
BS = chr(92)

PREAMBLE = r"""\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}% Math and Physical Sciences Numbered Reference Style

\usepackage{graphicx}%
\usepackage{multirow}%
\usepackage{amsmath,amssymb,amsfonts}%
\usepackage{amsthm}%
\usepackage{mathrsfs}%
\usepackage[title]{appendix}%
\usepackage{xcolor}%
\usepackage{textcomp}%
\usepackage{manyfoot}%
\usepackage{booktabs}%

\raggedbottom

\begin{document}
"""

FRONT = r"""
\author*[1]{\fnm{Quang-Vinh} \sur{Dang}}\email{vinh.dq4@buv.edu.vn}
\author[2]{\fnm{Dat} \sur{Le}}
\author[3]{\fnm{Minh Ngoc} \sur{Dinh}}

\affil*[1]{\orgname{British University Vietnam}, \orgaddress{\city{Hung Yen}, \country{Vietnam}}}
\affil[2]{\orgname{Ho Chi Minh City University of Economics and Finance}, \orgaddress{\city{Ho Chi Minh City}, \country{Vietnam}}}
\affil[3]{\orgname{Millennia Education}, \orgaddress{\city{Ho Chi Minh City}, \country{Vietnam}}}
"""

BACK = r"""
\backmatter

\section*{Declarations}

\begin{itemize}
\item Funding: The authors received no funding for this work.
\item Conflict of interest/Competing interests: The authors declare that they have no competing interests.
\item Ethics approval and consent to participate: Not applicable (public datasets, no human participants).
\item Consent for publication: Not applicable.
\item Availability of data and materials: %%DATAAVAIL%%
\item Code availability: \url{https://github.com/vinhqdang/cross_ids}.
\item Author contribution: Quang-Vinh Dang conceived the study, designed the methods, wrote the software, ran the experiments and wrote the manuscript. Dat Le contributed to the software, the data curation and the validation of the results. Minh Ngoc Dinh contributed to the statistical analysis and reviewed and edited the manuscript. All authors read and approved the final manuscript.
\end{itemize}

\bibliography{refs}

\end{document}
"""

RESIZEBOX = re.compile(
    re.escape(BS + "resizebox") + r"\{[^}]*\}\{!\}\{(" + re.escape(BS + "begin{tabular}") + r".*?" + re.escape(BS + "end{tabular}") + r")\}",
    re.S,
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf8").replace("\r\n", "\n")


def shrink_tables(body: str) -> str:
    """The sn-jnl table environment does not accept \\resizebox around a tabular: use a smaller font instead."""

    def repl(m: re.Match) -> str:
        spec = re.match(re.escape(BS + "begin{tabular}") + r"\{([^\n]*?)\}\n", m.group(1)).group(1)
        columns = len(re.findall(r"[lcrp]", re.sub(r"p\{[^}]*\}", "p", spec)))
        size, sep = (("tiny", "1.2pt") if columns >= 14 else ("tiny", "1.8pt") if columns >= 7
                     else ("scriptsize", "2.5pt") if columns >= 5 else ("footnotesize", "4pt"))
        return "{" + BS + size + BS + "setlength{" + BS + "tabcolsep}{" + sep + "}" + m.group(1) + "}"

    return RESIZEBOX.sub(repl, body)


TABULAR = re.compile(re.escape(BS + "begin{tabular}") + r"\{([^\n]*?)\}\n(.*?)" + re.escape(BS + "end{tabular}"), re.S)
PCOL = re.compile(r"p\{([0-9.]+)" + re.escape(BS + "textwidth") + r"\}")


def fit_tables(body: str) -> str:
    """Tables that were not shrunk yet: smaller font for five or more columns, narrower paragraph columns."""

    def repl(m: re.Match) -> str:
        before = body[max(0, m.start() - 40):m.start()]
        if "tabcolsep}{" in before:
            return m.group(0)
        spec = PCOL.sub(lambda p: "p{" + f"{float(p.group(1)) * 0.93:.3f}" + BS + "textwidth}", m.group(1))
        columns = len(re.findall(r"[lcrp]", re.sub(r"p\{[^}]*\}", "p", spec)))
        block = BS + "begin{tabular}{" + spec + "}\n" + m.group(2) + BS + "end{tabular}"
        if columns >= 5:
            return "{" + BS + "scriptsize" + BS + "setlength{" + BS + "tabcolsep}{2.5pt}" + block + "}"
        return block

    return TABULAR.sub(repl, body)


def main() -> None:
    src = read(PAPER / "ids.tex")
    title = re.search(re.escape(BS + "title{") + r"(.*?)\}\n", src, re.S).group(1)
    abstract = re.search(re.escape(BS + "begin{abstract}") + r"\n(.*?)\n" + re.escape(BS + "end{abstract}"), src, re.S).group(1)
    abstract = " ".join(line.strip() for line in abstract.splitlines() if line.strip())
    a = src.index(BS + "section{Introduction}")
    b = src.index(BS + "section*{Availability of data and materials}") if (BS + "section*{Availability of data and materials}") in src else src.index(BS + "bibliographystyle")
    body = src[a:b].rstrip("\n") + "\n"

    def inline(m: re.Match) -> str:
        return read(PAPER / "tables" / f"{m.group(1)}.tex").strip()

    body = re.sub(re.escape(BS + "input{tables/") + r"([A-Za-z0-9_]+)\}", inline, body)
    assert BS + "input{" not in body, "an input command was left in the body"
    body = fit_tables(shrink_tables(body))
    body = body.replace(BS + "begin{table}[t]", BS + "begin{table}[htbp]")
    keywords = ("network intrusion detection, cross-dataset generalisation, domain shift, benchmark, negative results, "
                "few-shot learning, tabular foundation model")
    text = (PREAMBLE
            + BS + "title[" + "Transfer in cross-dataset intrusion detection" + "]{" + title + "}\n"
            + FRONT
            + "\n" + BS + "abstract{" + abstract + "}\n"
            + "\n" + BS + "keywords{" + keywords + "}\n"
            + "\n" + BS + "maketitle\n\n"
            + body
            + BACK.replace("%%DATAAVAIL%%", read(PAPER / "dataavail.tex").strip().replace("\n", " ")))
    OUT.mkdir(exist_ok=True)
    (OUT / "ids_sn.tex").write_text(text, encoding="utf8")
    shutil.copyfile(PAPER / "refs.bib", OUT / "refs.bib")
    shutil.copytree(PAPER / "figures", OUT / "figures", dirs_exist_ok=True)
    print("wrote", OUT / "ids_sn.tex", len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
