"""
Command-Line Interface for NetReach v0.7.0.

Commands:
- netreach audit: Audit lateral movement reachability and neural path trace across recurrent steps T.
- netreach harden: Identify critical attack corridor chokepoints and generate actionable remediations.
- netreach view: Generate an interactive Cytoscape.js corridor visualization and edge severance simulation.
- netreach serve: Launch headless REST API daemon for SIEM/SOAR and enterprise platform integration.
- netreach benchmark: Run automated latency, parameter audit, and accuracy benchmarks.
"""

import os
import sys
import json
import argparse
from typing import List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from netreach.engine import NetReachEngine
from netreach.remediation import generate_markdown_report

console = Console()


def print_banner():
    banner = Text()
    banner.append("⚡ NetReach ", style="bold cyan")
    banner.append("v0.7.0 — Enterprise Neural Attack Path & Remediation Engine\n", style="bold white")
    banner.append("Powered by Bidirectional Recurrent Latent Scaling (38M Params, INT8 Optimized)", style="dim")
    console.print(Panel(banner, border_style="cyan"))


def cmd_audit(args):
    """Execute attack path reachability audit and neural path tracing."""
    if not args.json:
        print_banner()

    depths = tuple(args.depth) if args.depth else (4, 8, 12, 16)

    try:
        if not args.json:
            with console.status("[bold green]Loading neural reachability engine (v4 38M)..."):
                engine = NetReachEngine(
                    checkpoint_path=args.checkpoint,
                    device=args.device,
                    use_int8=args.int8,
                )
            with console.status(f"[bold yellow]Extracting attack corridor from '{args.src}' to '{args.dst}'..."):
                result = engine.audit(
                    graph_source=args.graph,
                    source=args.src,
                    target=args.dst,
                    depths=depths,
                )
        else:
            engine = NetReachEngine(
                checkpoint_path=args.checkpoint,
                device=args.device,
                use_int8=args.int8,
            )
            result = engine.audit(
                graph_source=args.graph,
                source=args.src,
                target=args.dst,
                depths=depths,
            )
    except Exception as e:
        console.print(f"[bold red]Audit Error:[/bold red] {e}")
        sys.exit(1)

    if args.json:
        data = {
            "source": result.source,
            "target": result.target,
            "risk_level": result.risk_level,
            "probabilities": {str(k): round(v, 4) for k, v in result.probabilities.items()},
            "cosine_similarities": {str(k): round(v, 4) for k, v in result.cosine_similarities.items()},
            "ground_truth_path": result.ground_truth_path,
            "shortest_hop_count": result.shortest_hop_count,
            "latency_ms": round(result.inference_latency_ms, 2),
            "subgraph_nodes": result.subgraph_nodes,
            "subgraph_edges": result.subgraph_edges,
            "oversmoothed": result.oversmoothed,
            "bridge_node": result.bridge_node,
            "neural_path": result.neural_path,
            "path_relations": result.path_relations,
            "is_valid_walk": result.is_valid_walk,
            "act_steps": round(result.act_steps, 2),
            "peak_collision": round(result.peak_collision, 4),
        }
        print(json.dumps(data, indent=2))
        return

    # Visual Rich Presentation
    console.print(f"\n[bold white]Source Asset:[/bold white] [bold cyan]{result.source}[/bold cyan]")
    console.print(f"[bold white]Target Asset:[/bold white] [bold red]{result.target}[/bold red]")
    mode_str = f"INT8 Quantized CPU" if engine.use_int8 else f"{engine.device}"
    console.print(
        f"[dim]Corridor Subgraph: {result.subgraph_nodes} nodes, {result.subgraph_edges} edges | "
        f"Latency: {result.inference_latency_ms:.2f} ms ({mode_str}, {engine.param_count:,} params)[/dim]\n"
    )

    # Collision Peak & Halting Telemetry
    if result.bridge_node is not None:
        console.print(
            f"[bold white]Collision Bridge Node ($v^*$):[/bold white] [bold magenta]{result.bridge_node}[/bold magenta]"
        )
    valid_walk_str = "[bold green]VALID[/bold green]" if result.is_valid_walk else "[bold yellow]DISCONTINUOUS[/bold yellow]"
    console.print(
        f"[bold white]ACT Telemetry:[/bold white] Halting Steps = [bold cyan]{result.act_steps:.1f}[/bold cyan] | "
        f"Peak Collision Affinity = [bold cyan]{result.peak_collision:.3f}[/bold cyan] | "
        f"Beam Walk: {valid_walk_str}\n"
    )

    table = Table(
        title="Lateral Movement Reachability vs. Latent Compute Depth (T)",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Recurrent Steps (T)", justify="center", style="bold cyan")
    table.add_column("Reachability Probability", justify="center", style="bold")
    table.add_column("Pairwise Cosine Sim", justify="center", style="dim")
    table.add_column("Risk Assessment", justify="center", style="bold")

    for step in sorted(result.probabilities.keys()):
        prob = result.probabilities[step]
        cos_sim = result.cosine_similarities[step]

        if prob >= 0.75:
            risk_label = "[bold red]CRITICAL RISK[/bold red]"
            prob_str = f"[bold red]{prob*100:.1f}%[/bold red]"
        elif prob >= 0.50:
            risk_label = "[bold yellow]HIGH RISK[/bold yellow]"
            prob_str = f"[bold yellow]{prob*100:.1f}%[/bold yellow]"
        elif prob >= 0.30:
            risk_label = "[bold cyan]MEDIUM RISK[/bold cyan]"
            prob_str = f"[cyan]{prob*100:.1f}%[/cyan]"
        else:
            risk_label = "[bold green]LOW RISK / UNREACHABLE[/bold green]"
            prob_str = f"[green]{prob*100:.1f}%[/green]"

        table.add_row(f"T = {step:2d}", prob_str, f"{cos_sim:.3f}", risk_label)

    console.print(table)

    # Reconstructed Attack Path step-by-step
    if result.path_relations:
        console.print(f"\n[bold yellow]Reconstructed Attack Chain ({len(result.path_relations)} hops):[/bold yellow]")
        for idx, step in enumerate(result.path_relations, 1):
            u = step["source"]
            v = step["target"]
            ad_rel = step.get("ad_relation", step.get("relation", "AdminTo"))
            canon_rel = step.get("relation", "AdminTo")
            rel_label = f"{canon_rel} [{ad_rel}]" if ad_rel != canon_rel else canon_rel
            console.print(f"  [dim]{idx:2d}.[/dim] [bold cyan][{u}][/bold cyan] ──([bold yellow]{rel_label}[/bold yellow])──► [bold red][{v}][/bold red]")
        console.print()
    elif result.neural_path and len(result.neural_path) > 1:
        path_str = " [bold red]──►[/bold red] ".join([f"[cyan][{node}][/cyan]" for node in result.neural_path])
        console.print(f"\n[bold yellow]Neural Attack Path ({len(result.neural_path) - 1} hops):[/bold yellow]")
        console.print(f"  {path_str}\n")
    elif result.ground_truth_path:
        path_str = " [bold red]──►[/bold red] ".join([f"[cyan][{node}][/cyan]" for node in result.ground_truth_path])
        console.print(f"\n[bold yellow]Discovered Graph Path ({result.shortest_hop_count} hops):[/bold yellow]")
        console.print(f"  {path_str}\n")
    else:
        console.print(f"\n[bold green]No attack chain identified between specified endpoints.[/bold green]\n")

    # Risk Summary Alert Box
    panel_title = f"RISK ASSESSMENT: {result.risk_level}"
    if result.risk_level == "CRITICAL":
        alert_body = (
            f"[bold red]CRITICAL EXPOSURE:[/bold red] Exploitable multi-hop lateral movement chain identified ({result.probabilities[depths[-1]]*100:.1f}%).\n"
            f"Adversary compromising '{result.source}' can pivot through intermediate assets to compromise '{result.target}'."
        )
        panel_color = "red"
    elif result.risk_level == "HIGH":
        alert_body = (
            f"[bold yellow]HIGH RISK:[/bold yellow] Probable lateral movement corridor identified ({result.probabilities[depths[-1]]*100:.1f}%).\n"
            f"Review intermediate privilege escalation and administrative delegation relationships."
        )
        panel_color = "yellow"
    else:
        alert_body = (
            f"[bold green]SEGMENTED:[/bold green] No viable lateral movement chain identified ({result.probabilities[depths[-1]]*100:.1f}%).\n"
            f"Target appears segmented from source position."
        )
        panel_color = "green"

    console.print(Panel(alert_body, title=panel_title, border_style=panel_color))


def cmd_benchmark(args):
    """Run latency and accuracy benchmark."""
    if not args.json:
        print_banner()
        console.print("[bold cyan]Initializing NetReach Benchmark Suite...[/bold cyan]")
        engine = NetReachEngine(
            checkpoint_path=args.checkpoint,
            device=args.device,
            use_int8=args.int8,
        )
        mode_str = "INT8 CPU" if engine.use_int8 else str(engine.device)
        console.print(f"[dim]Device: {mode_str} | Parameter Budget: {engine.param_count:,} params[/dim]")
        with console.status(f"[bold yellow]Executing {args.samples} reachability samples across depths..."):
            bench = engine.benchmark(num_samples=args.samples)
    else:
        engine = NetReachEngine(
            checkpoint_path=args.checkpoint,
            device=args.device,
            use_int8=args.int8,
        )
        bench = engine.benchmark(num_samples=args.samples)

    if args.json:
        print(json.dumps(bench, indent=2))
        return

    # Print summary cards
    console.print("\n")
    metric_table = Table(title="NetReach Engine Performance Audit", header_style="bold cyan")
    metric_table.add_column("Metric", style="bold white")
    metric_table.add_column("Value", style="bold green", justify="right")

    metric_table.add_row("Compute Device", bench["device"])
    metric_table.add_row("Total Parameters", f"{bench['parameter_count']:,}")
    metric_table.add_row("Mean Latency", f"{bench['mean_latency_ms']} ms / sample")
    metric_table.add_row("P95 Latency", f"{bench['latency_p95_ms']} ms")
    metric_table.add_row("Evaluation Samples", str(bench["tested_samples"]))
    console.print(metric_table)

    console.print("\n")
    acc_summary = bench.get("accuracy_summary", {})
    if acc_summary:
        first_hop = next(iter(acc_summary.values()))
        depth_keys = sorted(first_hop.keys(), key=lambda x: int(x.replace("T=", "")))

        acc_table = Table(title="Accuracy vs. Hop Length (H) Across Recurrent Depths (T)", header_style="bold magenta")
        acc_table.add_column("Hops (H)", justify="center", style="cyan")
        for d_key in depth_keys:
            acc_table.add_column(d_key, justify="center", style="yellow")

        for hop_key, depth_accs in acc_summary.items():
            row_vals = [f"{depth_accs.get(d, 0.0)*100:.1f}%" for d in depth_keys]
            acc_table.add_row(hop_key, *row_vals)

        console.print(acc_table)
    console.print("\n[bold green]✔ Benchmark completed successfully.[/bold green]\n")


def cmd_harden(args):
    """Execute chokepoint analysis and generate remediations via neural edge ablation."""
    if not args.json:
        print_banner()

    try:
        if not args.json:
            with console.status("[bold green]Loading neural reachability engine (v4 38M)..."):
                engine = NetReachEngine(
                    checkpoint_path=args.checkpoint,
                    device=args.device,
                    use_int8=args.int8,
                )
            with console.status(f"[bold yellow]Performing neural edge ablation between '{args.src}' and '{args.dst}'..."):
                result = engine.harden(
                    graph_source=args.graph,
                    source=args.src,
                    target=args.dst,
                    top_k=args.top_k,
                    steps=args.depth,
                )
        else:
            engine = NetReachEngine(
                checkpoint_path=args.checkpoint,
                device=args.device,
                use_int8=args.int8,
            )
            result = engine.harden(
                graph_source=args.graph,
                source=args.src,
                target=args.dst,
                top_k=args.top_k,
                steps=args.depth,
            )
    except Exception as e:
        console.print(f"[bold red]Harden Error:[/bold red] {e}")
        sys.exit(1)

    if args.json:
        data = {
            "source": result.source,
            "target": result.target,
            "baseline_risk_level": result.baseline_risk_level,
            "baseline_probability": round(result.baseline_probability, 4),
            "act_steps": round(result.act_steps, 2),
            "bridge_node": result.bridge_node,
            "attack_path": result.attack_path,
            "path_relations": result.path_relations,
            "analysis_latency_ms": round(result.analysis_latency_ms, 2),
            "subgraph_nodes": result.subgraph_nodes,
            "subgraph_edges": result.subgraph_edges,
            "chokepoints": [
                {
                    "rank": cp.rank,
                    "source": cp.source,
                    "target": cp.target,
                    "relation": cp.relation,
                    "ad_relation": cp.ad_relation,
                    "baseline_prob": round(cp.baseline_prob, 4),
                    "ablated_prob": round(cp.ablated_prob, 4),
                    "delta_p": round(cp.delta_p, 4),
                    "mitigation_cmd": cp.mitigation_cmd,
                    "cypher_query": cp.cypher_query,
                    "description": cp.description,
                }
                for cp in result.chokepoints
            ],
        }
        print(json.dumps(data, indent=2))
        if args.export:
            md_content = generate_markdown_report(result)
            with open(args.export, "w", encoding="utf-8") as f:
                f.write(md_content)
        return

    # Visual Rich Presentation
    console.print(f"\n[bold white]Source Asset:[/bold white] [bold cyan]{result.source}[/bold cyan]")
    console.print(f"[bold white]Target Asset:[/bold white] [bold red]{result.target}[/bold red]")
    mode_str = "INT8 Quantized CPU" if engine.use_int8 else f"{engine.device}"
    console.print(
        f"[dim]Corridor Subgraph: {result.subgraph_nodes} nodes, {result.subgraph_edges} edges | "
        f"Ablation Latency: {result.analysis_latency_ms:.2f} ms ({mode_str}, {engine.param_count:,} params)[/dim]\n"
    )

    # Discovered Attack Path Display
    if result.path_relations:
        console.print(Panel(
            Text(f"Detected Lateral Movement Corridor ({len(result.path_relations)} Hops)", style="bold yellow"),
            border_style="yellow",
        ))
        for i, edge in enumerate(result.path_relations, 1):
            rel = edge.get("ad_relation", edge.get("relation", "AdminTo"))
            console.print(f"  [dim]{i}.[/dim] [bold cyan]{edge['source']}[/bold cyan] ──([magenta]{rel}[/magenta])──► [bold green]{edge['target']}[/bold green]")
        console.print()

    # Chokepoints Table
    if not result.chokepoints:
        console.print("[bold yellow]No critical chokepoints found along this path.[/bold yellow]\n")
    else:
        table = Table(
            title="Top Defensive Chokepoints (Ranked by ΔP Reachability Collapse)",
            header_style="bold magenta",
        )
        table.add_column("Rank", justify="center", style="bold white", width=6)
        table.add_column("Chokepoint Edge", style="bold cyan")
        table.add_column("Relation", justify="center", style="magenta")
        table.add_column("Ablated Reachability", justify="right", style="yellow")
        table.add_column("Impact (ΔP)", justify="right", style="bold green")
        table.add_column("Remediation Action", style="white")

        for cp in result.chokepoints:
            edge_str = f"{cp.source} -> {cp.target}"
            impact_str = f"{cp.delta_p * 100:+.1f}%"
            ablated_str = f"{cp.ablated_prob * 100:.1f}%"
            table.add_row(f"#{cp.rank}", edge_str, cp.ad_relation, ablated_str, impact_str, cp.description)

        console.print(table)
        console.print()

        # Detailed Remediation Playbook Panel for Top 1
        top_cp = result.chokepoints[0]
        rec_text = Text()
        rec_text.append("Recommended Single Action to Break Attack Path:\n", style="bold underline yellow")
        rec_text.append("Sever edge: ", style="bold")
        rec_text.append(f"[{top_cp.source}] --({top_cp.ad_relation})--> [{top_cp.target}]\n", style="bold cyan")
        rec_text.append("Reachability Drops: ", style="bold")
        rec_text.append(f"{top_cp.baseline_prob * 100:.1f}% ➔ {top_cp.ablated_prob * 100:.1f}% ({top_cp.delta_p * 100:+.1f}% collapse)\n\n", style="bold green")
        rec_text.append("PowerShell Mitigation:\n", style="bold magenta")
        rec_text.append(f"{top_cp.mitigation_cmd}\n\n", style="dim")
        rec_text.append("BloodHound / Cypher Simulation Query:\n", style="bold magenta")
        rec_text.append(f"{top_cp.cypher_query}", style="dim")

        console.print(Panel(rec_text, title="🛡️ Top Priority Remediation", border_style="green"))

    # Export markdown report if requested
    if args.export:
        md_content = generate_markdown_report(result)
        with open(args.export, "w", encoding="utf-8") as f:
            f.write(md_content)
        console.print(f"\n[bold green]✔ Detailed remediation report exported to '{args.export}'[/bold green]\n")


def cmd_view(args):
    """Generate an interactive HTML/JavaScript graph visualization of the corridor and chokepoints."""
    print_banner()

    try:
        with console.status("[bold green]Loading neural reachability engine (v4 38M)..."):
            engine = NetReachEngine(
                checkpoint_path=args.checkpoint,
                device=args.device,
                use_int8=args.int8,
            )
        with console.status(f"[bold yellow]Analyzing attack corridor & chokepoints between '{args.src}' and '{args.dst}'..."):
            harden_res = engine.harden(
                graph_source=args.graph,
                source=args.src,
                target=args.dst,
                top_k=args.top_k,
                steps=args.depth,
            )

        from netreach.adapters.graph_loader import load_graph
        graph = load_graph(args.graph)

        from netreach.visualizer import generate_corridor_visualization
        out_path = generate_corridor_visualization(
            harden_result=harden_res,
            graph=graph,
            output_path=args.output,
        )

        console.print(f"\n[bold green]✔ Interactive corridor visualization generated:[/bold green] [bold cyan]{out_path}[/bold cyan]")
        console.print(f"[dim]Open this file in any web browser to explore interactive node traversal and real-time edge severance simulations.[/dim]\n")

        if args.open:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(out_path)}")

    except Exception as e:
        console.print(f"[bold red]Visualization Error:[/bold red] {e}")
        sys.exit(1)


def cmd_serve(args):
    """Launch NetReach REST API daemon."""
    print_banner()
    try:
        from netreach.server import run_server
        console.print(f"[bold green]Starting NetReach REST API Daemon on [cyan]http://{args.host}:{args.port}[/cyan]...[/bold green]")
        console.print(f"[dim]Endpoints available: POST /api/v1/audit, POST /api/v1/harden, GET /api/v1/health[/dim]\n")
        run_server(
            host=args.host,
            port=args.port,
            checkpoint_path=args.checkpoint,
            device=args.device,
            use_int8=args.int8,
        )
    except ImportError as e:
        console.print(f"[bold red]Dependency Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Server Error:[/bold red] {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="netreach",
        description="NetReach: Neural Network Security Reachability & Lateral Movement Engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'audit' command
    audit_parser = subparsers.add_parser("audit", help="Audit attack path reachability between two nodes.")
    audit_parser.add_argument("--graph", "-g", type=str, required=True, help="Path to graph file (JSON, GraphML, DOT).")
    audit_parser.add_argument("--src", "-s", type=str, required=True, help="Source entity (e.g., Workstation-01).")
    audit_parser.add_argument("--dst", "-d", type=str, required=True, help="Target entity (e.g., Domain-Controller).")
    audit_parser.add_argument("--depth", type=int, nargs="+", default=[4, 8, 12, 16], help="Recurrent test steps T.")
    audit_parser.add_argument("--device", type=str, default=None, help="Force device ('cpu' or 'cuda').")
    audit_parser.add_argument("--checkpoint", type=str, default=None, help="Custom weights file path.")
    audit_parser.add_argument("--int8", action="store_true", help="Enable dynamic INT8 quantization for fast CPU inference.")
    audit_parser.add_argument("--json", action="store_true", help="Output raw JSON results.")

    # 'harden' command
    harden_parser = subparsers.add_parser("harden", help="Identify critical attack corridor chokepoints and generate remediations.")
    harden_parser.add_argument("--graph", "-g", type=str, required=True, help="Path to graph file (JSON, GraphML, DOT).")
    harden_parser.add_argument("--src", "-s", type=str, required=True, help="Source entity (e.g., Workstation-01).")
    harden_parser.add_argument("--dst", "-d", type=str, required=True, help="Target entity (e.g., Domain-Controller).")
    harden_parser.add_argument("--depth", type=int, default=None, help="Recurrent evaluation steps T.")
    harden_parser.add_argument("--export", "-e", type=str, default=None, help="Path to export Markdown report.")
    harden_parser.add_argument("--top-k", "-k", type=int, default=3, help="Number of highest-impact chokepoints to return.")
    harden_parser.add_argument("--device", type=str, default=None, help="Force device ('cpu' or 'cuda').")
    harden_parser.add_argument("--checkpoint", type=str, default=None, help="Custom weights file path.")
    harden_parser.add_argument("--int8", action="store_true", help="Enable dynamic INT8 quantization for fast CPU inference.")
    harden_parser.add_argument("--json", action="store_true", help="Output raw JSON results.")

    # 'view' command
    view_parser = subparsers.add_parser("view", help="Generate an interactive HTML/JavaScript graph visualization.")
    view_parser.add_argument("--graph", "-g", type=str, required=True, help="Path to graph file (JSON, GraphML, DOT).")
    view_parser.add_argument("--src", "-s", type=str, required=True, help="Source entity (e.g., Workstation-01).")
    view_parser.add_argument("--dst", "-d", type=str, required=True, help="Target entity (e.g., Domain-Controller).")
    view_parser.add_argument("--output", "-o", type=str, default="corridor.html", help="Path to output HTML file.")
    view_parser.add_argument("--depth", type=int, default=None, help="Recurrent evaluation steps T.")
    view_parser.add_argument("--top-k", "-k", type=int, default=3, help="Number of highest-impact chokepoints to display.")
    view_parser.add_argument("--device", type=str, default=None, help="Force device ('cpu' or 'cuda').")
    view_parser.add_argument("--checkpoint", type=str, default=None, help="Custom weights file path.")
    view_parser.add_argument("--int8", action="store_true", help="Enable dynamic INT8 quantization for fast CPU inference.")
    view_parser.add_argument("--open", action="store_true", help="Open generated visualization directly in web browser.")

    # 'serve' command
    serve_parser = subparsers.add_parser("serve", help="Launch headless REST API daemon.")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1).")
    serve_parser.add_argument("--port", "-p", type=int, default=8080, help="Port to listen on (default: 8080).")
    serve_parser.add_argument("--device", type=str, default=None, help="Force device ('cpu' or 'cuda').")
    serve_parser.add_argument("--checkpoint", type=str, default=None, help="Custom weights file path.")
    serve_parser.add_argument("--int8", action="store_true", help="Enable dynamic INT8 quantization for fast CPU inference.")

    # 'benchmark' command
    bench_parser = subparsers.add_parser("benchmark", help="Run latency and accuracy benchmark.")
    bench_parser.add_argument("--samples", type=int, default=16, help="Number of benchmark samples.")
    bench_parser.add_argument("--device", type=str, default=None, help="Force device ('cpu' or 'cuda').")
    bench_parser.add_argument("--checkpoint", type=str, default=None, help="Custom weights file path.")
    bench_parser.add_argument("--int8", action="store_true", help="Enable dynamic INT8 quantization for fast CPU inference.")
    bench_parser.add_argument("--json", action="store_true", help="Output raw JSON results.")

    args = parser.parse_args()

    if args.command == "audit":
        cmd_audit(args)
    elif args.command == "harden":
        cmd_harden(args)
    elif args.command == "view":
        cmd_view(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
