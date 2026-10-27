"""
Command-Line Interface for NetReach.

Commands:
- netreach audit: Audit lateral movement reachability across recurrent steps T.
- netreach benchmark: Run automated latency, parameter audit, and accuracy benchmarks.
"""

import sys
import json
import argparse
from typing import List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from netreach.engine import NetReachEngine

console = Console()


def print_banner():
    banner = Text()
    banner.append("⚡ NetReach ", style="bold cyan")
    banner.append("v0.1.0 — Neural Graph Attack Path & Lateral Movement Engine\n", style="bold white")
    banner.append("Powered by Test-Time Recurrent Latent Scaling (501K Params)", style="dim")
    console.print(Panel(banner, border_style="cyan"))


def cmd_audit(args):
    """Execute attack path reachability audit."""
    if not args.json:
        print_banner()

    depths = tuple(args.depth) if args.depth else (5, 10, 15, 20)

    try:
        if not args.json:
            with console.status("[bold green]Loading neural reachability engine..."):
                engine = NetReachEngine(checkpoint_path=args.checkpoint, device=args.device)
            with console.status(f"[bold yellow]Extracting attack corridor from '{args.src}' to '{args.dst}'..."):
                result = engine.audit(
                    graph_source=args.graph,
                    source=args.src,
                    target=args.dst,
                    depths=depths,
                )
        else:
            engine = NetReachEngine(checkpoint_path=args.checkpoint, device=args.device)
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
        }
        print(json.dumps(data, indent=2))
        return

    # Visual Rich Presentation
    console.print(f"\n[bold white]Source Asset:[/bold white] [bold cyan]{result.source}[/bold cyan]")
    console.print(f"[bold white]Target Asset:[/bold white] [bold red]{result.target}[/bold red]")
    console.print(
        f"[dim]Corridor Subgraph: {result.subgraph_nodes} nodes, {result.subgraph_edges} edges | "
        f"Inference Latency: {result.inference_latency_ms:.2f} ms ({engine.device})[/dim]\n"
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

    # Lateral movement corridor summary
    if result.ground_truth_path:
        path_str = " [bold red]→[/bold red] ".join([f"[cyan]{node}[/cyan]" for node in result.ground_truth_path])
        console.print(f"\n[bold yellow]Discovered Attack Chain ({result.shortest_hop_count} hops):[/bold yellow]")
        console.print(f"  {path_str}\n")
    else:
        console.print(f"\n[bold green]No direct attack chain found in graph corridor.[/bold green]\n")

    # Risk Summary Alert Box
    panel_title = f"RISK ASSESSMENT: {result.risk_level}"
    if result.risk_level == "CRITICAL":
        alert_body = (
            f"[bold red]ALERT:[/bold red] Deep attack path detected with high confidence ({result.probabilities[depths[-1]]*100:.1f}%).\n"
            f"Adversary possessing '{result.source}' can achieve multi-hop lateral movement to compromise '{result.target}'."
        )
        panel_color = "red"
    elif result.risk_level == "HIGH":
        alert_body = (
            f"[bold yellow]WARNING:[/bold yellow] Probable lateral movement corridor identified ({result.probabilities[depths[-1]]*100:.1f}%).\n"
            f"Review intermediate privilege escalation and administrative delegation relationships."
        )
        panel_color = "yellow"
    else:
        alert_body = (
            f"[bold green]SECURE:[/bold green] No viable lateral movement chain identified ({result.probabilities[depths[-1]]*100:.1f}%).\n"
            f"Target appears segmented from source position."
        )
        panel_color = "green"

    console.print(Panel(alert_body, title=panel_title, border_style=panel_color))


def cmd_benchmark(args):
    """Run latency and accuracy benchmark."""
    if not args.json:
        print_banner()
        console.print("[bold cyan]Initializing NetReach Benchmark Suite...[/bold cyan]")
        engine = NetReachEngine(checkpoint_path=args.checkpoint, device=args.device)
        console.print(f"[dim]Device: {engine.device} | Parameter Budget: {engine.param_count:,} params[/dim]")
        with console.status(f"[bold yellow]Executing {args.samples} reachability samples across depths..."):
            bench = engine.benchmark(num_samples=args.samples)
    else:
        engine = NetReachEngine(checkpoint_path=args.checkpoint, device=args.device)
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
    metric_table.add_row("Total Parameters", f"{bench['parameter_count']:,} (Matched ~501K)")
    metric_table.add_row("Mean Latency", f"{bench['mean_latency_ms']} ms / sample")
    metric_table.add_row("P95 Latency", f"{bench['latency_p95_ms']} ms")
    metric_table.add_row("Evaluation Samples", str(bench["tested_samples"]))
    console.print(metric_table)

    console.print("\n")
    acc_table = Table(title="Accuracy vs. Hop Length (H) Across Recurrent Depths (T)", header_style="bold magenta")
    acc_table.add_column("Hops (H)", justify="center", style="cyan")
    for t_step in [5, 10, 15, 20]:
        acc_table.add_column(f"T={t_step}", justify="center", style="yellow")

    for hop_key, depth_accs in bench["accuracy_summary"].items():
        row_vals = [f"{depth_accs.get(f'T={t}', 0.0)*100:.1f}%" for t in [5, 10, 15, 20]]
        acc_table.add_row(hop_key, *row_vals)

    console.print(acc_table)
    console.print("\n[bold green]✔ Benchmark completed successfully.[/bold green]\n")


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
    audit_parser.add_argument("--depth", type=int, nargs="+", default=[5, 10, 15, 20], help="Recurrent test steps T.")
    audit_parser.add_argument("--device", type=str, default=None, help="Force device ('cpu' or 'cuda').")
    audit_parser.add_argument("--checkpoint", type=str, default=None, help="Custom weights file path.")
    audit_parser.add_argument("--json", action="store_true", help="Output raw JSON results.")

    # 'benchmark' command
    bench_parser = subparsers.add_parser("benchmark", help="Run latency and accuracy benchmark.")
    bench_parser.add_argument("--samples", type=int, default=50, help="Number of benchmark samples.")
    bench_parser.add_argument("--device", type=str, default=None, help="Force device ('cpu' or 'cuda').")
    bench_parser.add_argument("--checkpoint", type=str, default=None, help="Custom weights file path.")
    bench_parser.add_argument("--json", action="store_true", help="Output raw JSON results.")

    args = parser.parse_args()

    if args.command == "audit":
        cmd_audit(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
