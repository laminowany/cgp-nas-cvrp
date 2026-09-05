from graphviz import Digraph


def gene_type_name(gene_type):
    names = {
        1: "Identity",
        2: "Norm",
        3: "Linear",
        4: "Attention",
        5: "Add",
        6: "GELU",
        7: "ReLU",
    }
    return names.get(gene_type, f"Unknown({gene_type})")


def gene_color(gene_type):
    colors = {
        1: "lightgray",
        2: "lightblue",
        3: "orange",
        4: "red",
        5: "green",
        6: "yellow",
        7: "pink",
    }
    return colors.get(gene_type, "white")


def export_cgp_to_graphviz(
    genes,
    opts,
    filename,
    only_active,
    paper_style=False,
):
    rows = opts.y_dim
    dot = Digraph()

    dot.attr(
        rankdir="LR",
        splines="true",
        nodesep="0.8",
        ranksep="0.35",
        dpi="300",
    )

    dot.attr(
        "node",
        fontsize="28",
        fontname="Arial",
        penwidth="2.0",
    )

    dot.attr(
        "edge",
        penwidth="1.5",
        arrowsize="0.5",
    )

    dx = 2.5
    dy = -1.8

    visible_genes = [
        g
        for g in genes
        if g is not None and (not only_active or g.active)
    ]

    if paper_style:
        dot.node(
            "0",
            label="INPUT",
            style="dashed",
            fillcolor="white",
            width="1.8",
            height="1.8",
            fixedsize="true",
        )

    for g in visible_genes:
        col = g.pos // rows
        row = g.pos % rows

        x = col * dx
        y = row * dy

        style = "filled" if g.active else "dashed"

        label = gene_type_name(g.type)

        if g.type == 3 and g.args:
            if g.args[0] == 1:
                label += "\nUP"
            elif g.args[0] == -1:
                label += "\nDOWN"
            else:
                label += "\n-"

        if not paper_style:
            label = f"{g.pos}\n{label}"

        dot.node(
            str(g.pos),
            label=label,
            style=style,
            fillcolor=gene_color(g.type),
            pos=f"{x},{y}!",
            pin="true",
            width="1.8",
            height="1.8",
            fixedsize="true",
        )

    for g in visible_genes:
        for inp in g.inputs:
            dot.edge(str(inp), str(g.pos))
    if paper_style and visible_genes:
        last_gene = max(visible_genes, key=lambda g: g.pos)

        dot.node(
            "output",
            label="OUTPUT",
            style="dashed",
            fillcolor="white",
            width="1.8",
            height="1.8",
            fixedsize="true",
        )

        dot.edge(str(last_gene.pos), "output")

    dot.render(filename, format="png", cleanup=True)