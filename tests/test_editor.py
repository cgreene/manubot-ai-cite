from pathlib import Path
from unittest import mock

import pytest

from manubot.ai_editor import env_vars
from manubot.ai_editor.editor import ManuscriptEditor
from manubot.ai_editor.models import GPT3CompletionModel, RandomManuscriptRevisionModel

MANUSCRIPTS_DIR = Path(__file__).parent / "manuscripts"


def _check_nonparagraph_lines_are_preserved(input_filepath, output_filepath):
    """
    Checks whether non-paragraph lines in the input file are preserved in the output file.
    Non-paragraph lines are those that represent section headers, comments, blank lines, as
    defined in function ManuscriptEditor.line_is_not_part_of_paragraph.
    """
    # read lines from input file
    filepath = input_filepath
    assert filepath.exists()
    with open(filepath, "r") as infile:
        input_nonpar_lines = [
            line.rstrip()
            for line in infile
            if ManuscriptEditor.line_is_not_part_of_paragraph(line)
        ]

    # read lines from output file
    filepath = output_filepath
    assert filepath.exists()
    with open(filepath, "r") as infile:
        output_nonpar_lines = [
            line.rstrip()
            for line in infile
            if ManuscriptEditor.line_is_not_part_of_paragraph(line)
        ]

    # make sure all lines that are not "paragraphs" are preserved
    for input_line in input_nonpar_lines:
        # make sure there are nonparagraph lines left in output file
        assert (
            len(output_nonpar_lines) > 0
        ), "Output file has less non-paragraph lines than input file"

        # make sure nonparagraph lines are in the same order
        assert (
            input_line == output_nonpar_lines[0]
        ), f"{input_line} != {output_nonpar_lines[0]}"

        output_nonpar_lines.remove(input_line)

    # if nonparagraph lines were preserved, then the output_nonpar_liens should be empty
    assert (
        len(output_nonpar_lines) == 0
    ), f"Non-paragraph lines are not the same: {output_nonpar_lines}"


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_abstract(tmp_path, model):
    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("01.abstract.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "01.abstract.md",
        output_filepath=tmp_path / "01.abstract.md",
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_introduction(tmp_path, model):
    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("02.introduction.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "02.introduction.md",
        output_filepath=tmp_path / "02.introduction.md",
    )


def test_get_section_from_filename():
    assert ManuscriptEditor.get_section_from_filename("00.front-matter.md") is None
    assert ManuscriptEditor.get_section_from_filename("01.abstract.md") == "abstract"
    assert (
        ManuscriptEditor.get_section_from_filename("02.introduction.md")
        == "introduction"
    )
    assert ManuscriptEditor.get_section_from_filename("02.intro.md") is None
    assert ManuscriptEditor.get_section_from_filename("03.results.md") == "results"
    assert (
        ManuscriptEditor.get_section_from_filename("04.10.results_comp.md") == "results"
    )
    assert (
        ManuscriptEditor.get_section_from_filename("04.discussion.md") == "discussion"
    )
    assert (
        ManuscriptEditor.get_section_from_filename("05.conclusions.md") == "conclusions"
    )
    assert (
        ManuscriptEditor.get_section_from_filename("08.01.methods.ccc.md") == "methods"
    )
    assert (
        ManuscriptEditor.get_section_from_filename("08.15.methods.giant.md")
        == "methods"
    )
    assert ManuscriptEditor.get_section_from_filename("07.references.md") is None
    assert ManuscriptEditor.get_section_from_filename("06.acknowledgements.md") is None
    assert (
        ManuscriptEditor.get_section_from_filename("08.supplementary.md")
        == "supplementary material"
    )


@mock.patch.dict(
    "os.environ",
    {
        env_vars.SECTIONS_MAPPING: r"""
    {"02.intro.md": "introduction"}
    """
    },
)
def test_get_section_from_filename_using_environment_variable():
    assert (
        ManuscriptEditor.get_section_from_filename("02.introduction.md")
        == "introduction"
    )
    assert ManuscriptEditor.get_section_from_filename("02.intro.md") == "introduction"


@mock.patch.dict(
    "os.environ",
    {
        env_vars.SECTIONS_MAPPING: r"""
    {"02.intro.md": }
    """
    },
)
def test_get_section_from_filename_using_environment_variable_is_invalid():
    assert (
        ManuscriptEditor.get_section_from_filename("02.introduction.md")
        == "introduction"
    )
    assert ManuscriptEditor.get_section_from_filename("02.intro.md") is None


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_results_with_header_only(tmp_path, model):
    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("04.00.results.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "04.00.results.md",
        output_filepath=tmp_path / "04.00.results.md",
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_results_intro_with_figure(tmp_path, model):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("04.05.results_intro.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "04.05.results_intro.md",
        output_filepath=tmp_path / "04.05.results_intro.md",
    )

    # make sure the "image paragraph" was exactly copied to the output file
    assert (
        r"""
![
**Different types of relationships in data.**
Each panel contains a set of simulated data points described by two generic variables: $x$ and $y$.
The first row shows Anscombe's quartet with four different datasets (from Anscombe I to IV) and 11 data points each.
The second row contains a set of general patterns with 100 data points each.
Each panel shows the correlation value using Pearson ($p$), Spearman ($s$) and CCC ($c$).
Vertical and horizontal red lines show how CCC clustered data points using $x$ and $y$.
](images/intro/relationships.svg "Different types of relationships in data"){#fig:datasets_rel width="100%"}
    """.strip()
        in open(tmp_path / "04.05.results_intro.md").read()
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_methods_with_equation(tmp_path, model):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "phenoplier",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("07.00.methods.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "phenoplier" / "07.00.methods.md",
        output_filepath=tmp_path / "07.00.methods.md",
    )

    # make sure the "image paragraph" was exactly copied to the output file
    assert (
        r"""
$$
\mathbf{y} = \mathbf{t}_l \gamma_l + \bm{\epsilon}_l,
$$ {#eq:predixcan}
    """.strip()
        in open(tmp_path / "07.00.methods.md").read()
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_supplementary_material_with_tables_and_multiline_html_comments(
    tmp_path, model
):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("20.00.supplementary_material.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "20.00.supplementary_material.md",
        output_filepath=tmp_path / "20.00.supplementary_material.md",
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
| | **Interaction confidence** <!-- $colspan="7" -->    | | | | | | |
|:------:|:-----:|:-----:|:-----:|:--------:|:-----:|:-----:|:-----:|
| | **Blood** <!-- $colspan="3" --> | | | **Predicted cell type** <!-- $colspan="4" --> | | | |
| **Gene** |  **Min.** | **Avg.** | **Max.** |  **Cell type** | **Min.** | **Avg.** | **Max.** |
| *IFNG* | 0.19 | 0.42 | 0.54 | Natural killer cell<!-- $rowspan="2" --> | 0.74 | 0.90 | 0.99 |
| *SDS* | 0.18 | 0.29 | 0.41 | 0.65 | 0.81 | 0.94<!-- $removenext="2" --> |
| <!-- $colspan="7" --> |||||||
| *JUN* | 0.26 | 0.68 | 0.97 | Mononuclear phagocyte<!-- $rowspan="2" --> | 0.36 | 0.73 | 0.94 |
| *APOC1* | 0.22 | 0.47 | 0.77 | 0.29 | 0.50 | 0.80<!-- $removenext="2" --> |
| <!-- $colspan="7" --> |||||||
| *ZDHHC12* | 0.05 | 0.07 | 0.10 | Macrophage<!-- $rowspan="2" --> | 0.03 | 0.12 | 0.33 |
| *CCL18* | 0.74 | 0.79 | 0.86 | 0.36 | 0.70 | 0.90<!-- $removenext="2" --> |
| <!-- $colspan="7" --> |||||||
| *RASSF2* | 0.69 | 0.77 | 0.90 | Leukocyte<!-- $rowspan="2" --> | 0.66 | 0.74 | 0.88 |
| *CYTIP* | 0.74 | 0.85 | 0.91 | 0.76 | 0.84 | 0.91<!-- $removenext="2" --> |
| <!-- $colspan="7" --> |||||||
| *MYOZ1* | 0.09 | 0.17 | 0.37 | Skeletal muscle<!-- $rowspan="2" --> | 0.11 | 0.11 | 0.12 |
| *TNNI2* | 0.10 | 0.22 | 0.44 | 0.10 | 0.11 | 0.12<!-- $removenext="2" --> |
| <!-- $colspan="7" --> |||||||
| *PYGM* | 0.02 | 0.04 | 0.14 | Skeletal muscle<!-- $rowspan="2" --> | 0.01 | 0.02 | 0.04 |
| *TPM2* | 0.05 | 0.56 | 0.80 | 0.01 | 0.28 | 0.47<!-- $removenext="2" --> |

Table: Network statistics of six gene pairs shown in Figure @fig:upsetplot_coefs b for blood and predicted cell types.
Only gene pairs present in GIANT models are listed.
For each gene in the pair (first column), the minimum, average and maximum interaction coefficients with the other genes in the network are shown.
{#tbl:giant:weights}
    """.strip()
        in open(tmp_path / "20.00.supplementary_material.md").read()
    )

    # make sure the "HTML comment paragraph" was exactly copied to the output file
    assert (
        r"""
<!-- ![
**Predicted tissue-specific networks from GIANT for six gene pairs prioritized by correlation coefficients.**
Gene pairs are from Figure @fig:upsetplot_coefs b.
A node represents a gene and an edge the probability that two genes are part of the same biological process in a specific cell type.
The cell type for each gene network was automatically predicted using [@doi:10.1101/gr.155697.113], and it is indicated at the top-right corner of each network.
A maximum of 15 genes are shown for each subfigure.
The GIANT web application automatically determined a minimum interaction confidence (edges' weights) to be shown.
All these analyses can be performed online using the following links:
*IFNG* - *SDS* [@url:https://hb.flatironinstitute.org/gene/10993+3458],
*JUN* - *APOC1* [@url:https://hb.flatironinstitute.org/gene/3725+341],
*ZDHHC12* - *CCL18* [@url:https://hb.flatironinstitute.org/gene/6362+84885],
*RASSF2* - *CYTIP* [@url:https://hb.flatironinstitute.org/gene/9770+9595],
*MYOZ1* - *TNNI2* [@url:https://hb.flatironinstitute.org/gene/58529+7136],
*PYGM* - *TPM2* [@url:https://hb.flatironinstitute.org/gene/5837+7169].
The GIANT web-server was accessed on April 4, 2022.
](images/coefs_comp/giant_networks/auto_selected_tissues/main.svg "GIANT network interaction"){#fig:giant_gene_pairs:pred_tissue width="100%"} -->
    """.strip()
        in open(tmp_path / "20.00.supplementary_material.md").read()
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_supplementary_material_from_phenoplier_with_many_tables(
    tmp_path, model
):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "phenoplier",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("50.00.supplementary_material.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR
        / "phenoplier"
        / "50.00.supplementary_material.md",
        output_filepath=tmp_path / "50.00.supplementary_material.md",
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
<!-- LV603:multiplier_pathways:start -->
| Pathway                             | AUC   | FDR      |
|:------------------------------------|:------|:---------|
| IRIS Neutrophil-Resting             | 0.91  | 4.51e-35 |
| SVM Neutrophils                     | 0.98  | 1.43e-09 |
| PID IL8CXCR2 PATHWAY                | 0.81  | 7.04e-03 |
| SIG PIP3 SIGNALING IN B LYMPHOCYTES | 0.77  | 1.95e-02 |

Table: Pathways aligned to LV603 from the MultiPLIER models. {#tbl:sup:multiplier_pathways:lv603}
<!-- LV603:multiplier_pathways:end -->
    """.strip()
        in open(tmp_path / "50.00.supplementary_material.md").read()
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
<!-- LV603:phenomexcan_traits_assocs:start -->
| Trait description                         | Sample size   | Cases   | FDR            |
|:------------------------------------------|:--------------|:--------|:---------------|
| Basophill percentage                      | 349,861       |         | 1.19e&#8209;10 |
| Basophill count                           | 349,856       |         | 1.89e&#8209;05 |
| Treatment/medication code: ispaghula husk | 361,141       | 327     | 1.36e&#8209;02 |

Table: Significant trait associations of LV603 in PhenomeXcan. {#tbl:sup:phenomexcan_assocs:lv603}
<!-- LV603:phenomexcan_traits_assocs:end -->
    """.strip()
        in open(tmp_path / "50.00.supplementary_material.md").read()
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
<!-- LV603:emerge_traits_assocs:start -->
| Phecode                     | Trait description   | Sample size   | Cases   | FDR   |
|:----------------------------|:--------------------|:--------------|:--------|:------|
| No significant associations |                     |               |         |       |

Table: Significant trait associations of LV603 in eMERGE. {#tbl:sup:emerge_assocs:lv603}
<!-- LV603:emerge_traits_assocs:end -->
    """.strip()
        in open(tmp_path / "50.00.supplementary_material.md").read()
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_supplementary_material_from_phenoplier_with_many_tables_and_complex_html_comments(
    tmp_path, model
):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "phenoplier",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("50.01.supplementary_material.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR
        / "phenoplier"
        / "50.01.supplementary_material.md",
        output_filepath=tmp_path / "50.01.supplementary_material.md",
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
<!-- LV603:multiplier_pathways:start
this is a more complex multiline html comment -->
| Pathway                             | AUC   | FDR      |
|:------------------------------------|:------|:---------|
| IRIS Neutrophil-Resting             | 0.91  | 4.51e-35 |
| SVM Neutrophils                     | 0.98  | 1.43e-09 |
| PID IL8CXCR2 PATHWAY                | 0.81  | 7.04e-03 |
| SIG PIP3 SIGNALING IN B LYMPHOCYTES | 0.77  | 1.95e-02 |

Table: Pathways aligned to LV603 from the MultiPLIER models. {#tbl:sup:multiplier_pathways:lv603}
<!-- LV603:multiplier_pathways:end -->
    """.strip()
        in open(tmp_path / "50.01.supplementary_material.md").read()
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
<!-- LV603:phenomexcan_traits_assocs:start
and this html comments is multiline but

also has an empty line in the middle-->
| Trait description                         | Sample size   | Cases   | FDR            |
|:------------------------------------------|:--------------|:--------|:---------------|
| Basophill percentage                      | 349,861       |         | 1.19e&#8209;10 |
| Basophill count                           | 349,856       |         | 1.89e&#8209;05 |
| Treatment/medication code: ispaghula husk | 361,141       | 327     | 1.36e&#8209;02 |

Table: Significant trait associations of LV603 in PhenomeXcan. {#tbl:sup:phenomexcan_assocs:lv603}
<!-- LV603:phenomexcan_traits_assocs:end 


-->   
    """.strip()
        in open(tmp_path / "50.01.supplementary_material.md").read()
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
<!--

and this html multiline comment has a space
LV603:emerge_traits_assocs:start

-->
| Phecode                     | Trait description   | Sample size   | Cases   | FDR   |
|:----------------------------|:--------------------|:--------------|:--------|:------|
| No significant associations |                     |               |         |       |

Table: Significant trait associations of LV603 in eMERGE. {#tbl:sup:emerge_assocs:lv603}
<!-- LV603:emerge_traits_assocs:end -->
    """.strip()
        in open(tmp_path / "50.01.supplementary_material.md").read()
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_entire_manuscript(tmp_path, model):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    output_folder = tmp_path
    assert output_folder.exists()

    me.revise_manuscript(output_folder, model)

    output_md_files = list(output_folder.glob("*.md"))
    assert len(output_md_files) == 12
