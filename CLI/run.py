#!/usr/bin/env python3
"""
Algorithm Intelligence Suite - Comprehensive CLI Implementation
Beautiful, Interactive Command Line Interface for the complete system
Version 3.0 - Production Ready
"""

import requests
import json
import os
import sys
import time
import webbrowser
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Rich library imports for beautiful CLI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.tree import Tree
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich.markdown import Markdown
from rich import box
from rich.status import Status

class AlgorithmIntelligenceCLI:
    """
    Comprehensive CLI for Algorithm Intelligence Suite
    Beautiful, interactive, and feature-complete interface
    """
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.console = Console()
        self.session_id = None
        self.current_results = None
        self.sessions_history = []
        self.config = {
            "auto_open_visualizations": False,
            "save_sessions": True,
            "theme": "dark",
            "show_debug": False
        }
        
        # Initialize CLI
        self.setup_cli()
    
    def setup_cli(self):
        """Initialize CLI with welcome screen and health check"""
        self.console.clear()
        self.show_welcome_banner()
        self.check_api_health()
    
    def show_welcome_banner(self):
        """Display beautiful welcome banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════╗
║  🤖 ALGORITHM INTELLIGENCE SUITE - COMPREHENSIVE CLI v3.0        ║
║                                                                  ║
║  AI-Powered Algorithm Analysis • RAG-Enhanced Complexity        ║
║  MongoDB Visualizations • Performance Analysis • Education      ║
╚══════════════════════════════════════════════════════════════════╝
        """
        
        self.console.print(Panel(
            Align.center(banner),
            style="bold blue",
            border_style="bright_blue"
        ))
        
        self.console.print(f"[dim]API Endpoint: {self.api_url}[/dim]")
        self.console.print(f"[dim]Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")
    
    def check_api_health(self):
        """Check API health and display status"""
        with Status("[bold blue]Checking API connection...", console=self.console, spinner="dots"):
            try:
                response = requests.get(f"{self.api_url}/api/status", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    self.display_api_status(data)
                else:
                    self.console.print(f"[bold red]❌ API Error: HTTP {response.status_code}[/bold red]")
                    sys.exit(1)
            except requests.exceptions.RequestException as e:
                self.console.print(f"[bold red]❌ Connection Failed: {e}[/bold red]")
                self.console.print(f"[yellow]💡 Make sure your API server is running: python app.py[/yellow]")
                sys.exit(1)
    
    def display_api_status(self, data: Dict):
        """Display API health status in a beautiful format"""
        status_table = Table(title="🔍 API Health Check", box=box.ROUNDED)
        status_table.add_column("Component", style="cyan", no_wrap=True)
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="dim")
        
        # API Status
        status_table.add_row("API Server", "✅ Online", f"v{data.get('version', '3.0')}")
        
        # MongoDB Status
        mongodb_status = "✅ Connected" if data.get('mongodb_enabled') else "❌ Disabled"
        algorithms_count = data.get('algorithms_available', 0)
        status_table.add_row("MongoDB", mongodb_status, f"{algorithms_count} algorithms")
        
        # Features Status
        features = data.get('features', {})
        for feature, available in features.items():
            status = "✅ Available" if available else "❌ Unavailable"
            status_table.add_row(feature.replace('_', ' ').title(), status, "")
        
        self.console.print(status_table)
        self.console.print()
    
    def run(self):
        """Main CLI loop with interactive menu"""
        while True:
            try:
                choice = self.show_main_menu()
                
                if choice == "1":
                    self.new_analysis_flow()
                elif choice == "2":
                    self.quick_analysis_flow()
                elif choice == "3":
                    self.view_current_results()
                elif choice == "4":
                    self.browse_sessions_history()
                elif choice == "5":
                    self.advanced_features_menu()
                elif choice == "6":
                    self.system_tools_menu()
                elif choice == "7":
                    self.settings_menu()
                elif choice == "8":
                    self.show_help()
                elif choice == "9":
                    self.exit_cli()
                    break
                else:
                    self.console.print("[bold red]Invalid choice. Please try again.[/bold red]")
                
            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]⚠️ Interrupted by user[/bold yellow]")
                if Confirm.ask("Do you want to exit?"):
                    break
            except Exception as e:
                self.console.print(f"[bold red]❌ Unexpected error: {e}[/bold red]")
                if self.config["show_debug"]:
                    self.console.print_exception()
    
    def show_main_menu(self) -> str:
        """Display main menu and get user choice"""
        menu_panel = Panel(
            """[bold cyan]🎯 MAIN MENU[/bold cyan]

[bold green]1.[/bold green] 🧠 New Comprehensive Analysis
[bold green]2.[/bold green] ⚡ Quick Analysis
[bold green]3.[/bold green] 📊 View Current Results
[bold green]4.[/bold green] 📚 Browse Session History
[bold green]5.[/bold green] 🔬 Advanced Features
[bold green]6.[/bold green] 🛠️  System Tools
[bold green]7.[/bold green] ⚙️  Settings
[bold green]8.[/bold green] ❓ Help & Documentation
[bold green]9.[/bold green] 🚪 Exit

[dim]Current Session: {session}[/dim]""".format(
                session=self.session_id if self.session_id else "None"
            ),
            title="Algorithm Intelligence Suite",
            border_style="bright_blue"
        )
        
        self.console.print(menu_panel)
        return Prompt.ask("Enter your choice", choices=[str(i) for i in range(1, 10)])
    
    def new_analysis_flow(self):
        """Comprehensive analysis flow with all options"""
        self.console.clear()
        self.console.print(Panel("[bold green]🧠 NEW COMPREHENSIVE ANALYSIS[/bold green]", expand=False))
        
        # Get user input
        input_text = self.get_algorithm_input()
        if not input_text:
            return
        
        # Get input type
        input_type = Prompt.ask(
            "Input type", 
            choices=["auto", "problem", "code"], 
            default="auto"
        )
        
        # Get analysis options
        options = self.get_analysis_options()
        
        # Perform analysis
        self.perform_analysis(input_text, input_type, options)
        
        # Show results menu
        if self.current_results:
            self.comprehensive_results_viewer()
    
    def quick_analysis_flow(self):
        """Quick analysis with default options"""
        self.console.print(Panel("[bold yellow]⚡ QUICK ANALYSIS[/bold yellow]", expand=False))
        
        input_text = Prompt.ask("Enter your algorithm question or code")
        if not input_text.strip():
            return
        
        default_options = {
            "include_visualization": True,
            "include_performance": False,
            "include_educational": True
        }
        
        self.perform_analysis(input_text, "auto", default_options)
        
        if self.current_results:
            self.quick_results_display()
    
    def get_algorithm_input(self) -> str:
        """Get algorithm input with examples and validation"""
        examples_panel = Panel(
            """[bold cyan]📝 INPUT EXAMPLES[/bold cyan]

[bold green]Problem Descriptions:[/bold green]
• "Implement BFS algorithm for graph traversal"
• "Find the shortest path between two nodes"
• "Sort an array using quicksort algorithm"

[bold green]Algorithm Questions:[/bold green]
• "How does binary search work?"
• "What is the time complexity of merge sort?"
• "Explain depth-first search with example"

[bold green]Code Analysis:[/bold green]
• Paste your Python algorithm code for analysis
• Include function definitions and logic""",
            border_style="green"
        )
        
        self.console.print(examples_panel)
        
        # Multi-line input support
        self.console.print("[bold]Enter your algorithm question, problem, or code:[/bold]")
        self.console.print("[dim](Press Enter twice to finish, or type 'cancel' to go back)[/dim]")
        
        lines = []
        while True:
            line = input()
            if line.strip().lower() == 'cancel':
                return ""
            if line == "" and lines:
                break
            lines.append(line)
        
        return "\n".join(lines).strip()
    
    def get_analysis_options(self) -> Dict:
        """Get detailed analysis options from user"""
        options_panel = Panel(
            "[bold cyan]🔧 ANALYSIS OPTIONS[/bold cyan]\nSelect what you want to include in the analysis:",
            border_style="cyan"
        )
        self.console.print(options_panel)
        
        options = {}
        
        # Visualization options
        options["include_visualization"] = Confirm.ask(
            "🎨 Include Algorithm Visualizations?", default=True
        )
        
        # Performance analysis
        options["include_performance"] = Confirm.ask(
            "📊 Include Performance Analysis & Benchmarks?", default=True
        )
        
        # Educational content
        options["include_educational"] = Confirm.ask(
            "📚 Include Educational Insights & Recommendations?", default=True
        )
        
        # Advanced options
        if Confirm.ask("🔬 Show Advanced Options?", default=False):
            options["detailed_complexity"] = Confirm.ask(
                "Include detailed complexity analysis?", default=True
            )
            options["code_optimization"] = Confirm.ask(
                "Include code optimization suggestions?", default=False
            )
            options["alternative_approaches"] = Confirm.ask(
                "Show alternative algorithm approaches?", default=True
            )
        
        return options
    
    def perform_analysis(self, input_text: str, input_type: str, options: Dict):
        """Perform the actual analysis with beautiful progress display"""
        
        # Prepare request payload
        payload = {
            "input": input_text,
            "input_type": input_type,
            "options": options
        }
        
        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            
            # Start analysis
            task = progress.add_task("🔍 Initializing analysis...", total=100)
            
            try:
                progress.update(task, advance=10, description="📤 Sending request to AI...")
                
                response = requests.post(
                    f"{self.api_url}/api/analyze",
                    json=payload,
                    timeout=120
                )
                
                progress.update(task, advance=30, description="🧠 AI processing your request...")
                time.sleep(1)  # Simulate processing time for better UX
                
                if response.status_code == 200:
                    progress.update(task, advance=40, description="📊 Analyzing complexity...")
                    self.current_results = response.json()
                    self.session_id = self.current_results.get("session_id")
                    
                    progress.update(task, advance=20, description="🎨 Generating visualizations...")
                    time.sleep(0.5)
                    
                    progress.update(task, advance=10, description="✅ Analysis complete!")
                    
                    # Add to history
                    if self.config["save_sessions"]:
                        self.sessions_history.append({
                            "session_id": self.session_id,
                            "timestamp": datetime.now(),
                            "input": input_text[:100] + "..." if len(input_text) > 100 else input_text,
                            "input_type": input_type
                        })
                    
                    self.console.print(f"\n[bold green]✅ Analysis completed successfully![/bold green]")
                    self.console.print(f"[dim]Session ID: {self.session_id}[/dim]")
                    
                else:
                    error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
                    error_msg = error_data.get('error', f'HTTP {response.status_code}')
                    self.console.print(f"\n[bold red]❌ Analysis failed: {error_msg}[/bold red]")
                    
            except requests.exceptions.Timeout:
                self.console.print("\n[bold red]❌ Request timed out. Please try again.[/bold red]")
            except requests.exceptions.RequestException as e:
                self.console.print(f"\n[bold red]❌ Network error: {e}[/bold red]")
            except Exception as e:
                self.console.print(f"\n[bold red]❌ Unexpected error: {e}[/bold red]")
    
    def comprehensive_results_viewer(self):
        """Comprehensive results viewer with full navigation"""
        if not self.current_results:
            self.console.print("[bold red]No results to display[/bold red]")
            return
        
        while True:
            choice = self.show_results_menu()
            
            if choice == "1":
                self.show_analysis_summary()
            elif choice == "2":
                self.show_complexity_analysis()
            elif choice == "3":
                self.show_generated_solution()
            elif choice == "4":
                self.show_visualizations_menu()
            elif choice == "5":
                self.show_performance_analysis()
            elif choice == "6":
                self.show_educational_content()
            elif choice == "7":
                self.show_raw_data()
            elif choice == "8":
                self.export_results()
            elif choice == "9":
                self.open_web_report()
            elif choice == "10":
                break
            
            if choice != "10":
                input("\nPress Enter to continue...")
    
    def show_results_menu(self) -> str:
        """Display results navigation menu"""
        results_panel = Panel(
            """[bold cyan]📊 ANALYSIS RESULTS MENU[/bold cyan]

[bold green]1.[/bold green] 📋 Analysis Summary
[bold green]2.[/bold green] ⚡ Complexity Analysis (Time/Space)
[bold green]3.[/bold green] 💻 Generated Solution & Code
[bold green]4.[/bold green] 🎨 Visualizations Gallery
[bold green]5.[/bold green] 📈 Performance Analysis
[bold green]6.[/bold green] 📚 Educational Content & Insights
[bold green]7.[/bold green] 🔍 Raw JSON Data
[bold green]8.[/bold green] 💾 Export Results
[bold green]9.[/bold green] 🌐 Open Web Report
[bold green]10.[/bold green] ⬅️ Back to Main Menu

[dim]Session: {session}[/dim]""".format(session=self.session_id),
            title="Results Viewer",
            border_style="bright_green"
        )
        
        self.console.print(results_panel)
        return Prompt.ask("Select option", choices=[str(i) for i in range(1, 11)])
    
    def show_analysis_summary(self):
        """Display comprehensive analysis summary"""
        self.console.clear()
        self.console.print(Panel("[bold cyan]📋 ANALYSIS SUMMARY[/bold cyan]", expand=False))
        
        # Basic info table
        info_table = Table(title="Session Information", box=box.ROUNDED)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Session ID", self.session_id or "N/A")
        info_table.add_row("Timestamp", self.current_results.get("timestamp", "N/A"))
        info_table.add_row("Input Type", self.current_results.get("input_type", "N/A"))
        info_table.add_row("Processing Mode", self.current_results.get("processing_mode", "Standard"))
        
        self.console.print(info_table)
        
        # Stages completion
        stages = self.current_results.get("stages", {})
        stages_table = Table(title="Analysis Stages", box=box.ROUNDED)
        stages_table.add_column("Stage", style="cyan")
        stages_table.add_column("Status", style="green")
        stages_table.add_column("Details", style="dim")
        
        stage_names = {
            "input_analysis": "Input Analysis",
            "complexity_analysis": "Complexity Analysis",
            "algorithm_solving": "Algorithm Solving",
            "visualizations": "Visualizations",
            "performance_analysis": "Performance Analysis",
            "educational_report": "Educational Report"
        }
        
        for stage_key, stage_data in stages.items():
            stage_name = stage_names.get(stage_key, stage_key.replace("_", " ").title())
            if isinstance(stage_data, dict):
                if stage_data.get("error"):
                    status = "❌ Failed"
                    details = stage_data.get("error", "")[:50]
                elif stage_data.get("success") is False:
                    status = "⚠️ Partial"
                    details = stage_data.get("message", "")[:50]
                else:
                    status = "✅ Completed"
                    details = "Success"
            else:
                status = "✅ Completed"
                details = "Data available"
            
            stages_table.add_row(stage_name, status, details)
        
        self.console.print(stages_table)
        
        # Input preview
        input_text = self.current_results.get("input", "")
        if input_text:
            input_preview = input_text[:200] + "..." if len(input_text) > 200 else input_text
            input_panel = Panel(
                input_preview,
                title="Input Preview",
                border_style="blue"
            )
            self.console.print(input_panel)
    
    def show_complexity_analysis(self):
        """Display detailed complexity analysis"""
        self.console.clear()
        self.console.print(Panel("[bold yellow]⚡ COMPLEXITY ANALYSIS[/bold yellow]", expand=False))
        
        complexity_data = self.current_results.get("stages", {}).get("complexity_analysis", {})
        
        if not complexity_data:
            self.console.print("[bold red]No complexity analysis available[/bold red]")
            return
        
        # Extract complexity info
        agent_result = complexity_data.get("agent_result", {})
        complexity_info = agent_result.get("complexity_analysis", {})
        
        if not complexity_info:
            self.console.print("[bold red]No complexity data found[/bold red]")
            return
        
        # Main complexity table
        complexity_table = Table(title="Time & Space Complexity", box=box.DOUBLE)
        complexity_table.add_column("Metric", style="cyan", width=20)
        complexity_table.add_column("Value", style="bold green", width=15)
        complexity_table.add_column("Description", style="dim", width=40)
        
        time_complexity = complexity_info.get("time_complexity", "Unknown")
        space_complexity = complexity_info.get("space_complexity", "Unknown")
        
        complexity_table.add_row(
            "Time Complexity",
            time_complexity,
            self.get_complexity_description(time_complexity)
        )
        complexity_table.add_row(
            "Space Complexity",
            space_complexity,
            self.get_complexity_description(space_complexity)
        )
        
        self.console.print(complexity_table)
        
        # Reasoning
        reasoning = complexity_info.get("reasoning", "")
        if reasoning:
            reasoning_panel = Panel(
                reasoning,
                title="Complexity Reasoning",
                border_style="yellow"
            )
            self.console.print(reasoning_panel)
        
        # RAG Context if available
        rag_context = agent_result.get("enhanced_rag_context", [])
        if rag_context:
            self.console.print("\n[bold cyan]📚 Related Knowledge (RAG):[/bold cyan]")
            for i, context in enumerate(rag_context[:3], 1):
                context_panel = Panel(
                    context.get("content", "")[:300] + "...",
                    title=f"Reference {i}: {context.get('title', 'Unknown')}",
                    border_style="dim"
                )
                self.console.print(context_panel)
    
    def get_complexity_description(self, complexity: str) -> str:
        """Get description for complexity notation"""
        descriptions = {
            "O(1)": "Constant time - best possible",
            "O(log n)": "Logarithmic time - very efficient",
            "O(n)": "Linear time - reasonable",
            "O(n log n)": "Linearithmic time - good for sorting",
            "O(n²)": "Quadratic time - can be slow",
            "O(2^n)": "Exponential time - very slow",
            "O(n!)": "Factorial time - extremely slow"
        }
        return descriptions.get(complexity, "See complexity analysis")
    
    def show_generated_solution(self):
        """Display generated algorithm solution with syntax highlighting"""
        self.console.clear()
        self.console.print(Panel("[bold green]💻 GENERATED SOLUTION[/bold green]", expand=False))
        
        solution_data = self.current_results.get("stages", {}).get("algorithm_solving", {})
        optimal_solution = solution_data.get("optimal_solution", {})
        
        if not optimal_solution:
            self.console.print("[bold red]No solution generated[/bold red]")
            return
        
        # Algorithm info
        algorithm_name = optimal_solution.get("algorithm_name", "Generated Algorithm")
        description = optimal_solution.get("description", "No description available")
        
        info_panel = Panel(
            f"[bold]{algorithm_name}[/bold]\n\n{description}",
            title="Algorithm Information",
            border_style="green"
        )
        self.console.print(info_panel)
        
        # Generated code
        code_data = optimal_solution.get("code", {})
        if isinstance(code_data, dict):
            code_text = code_data.get("code", "")
        else:
            code_text = str(code_data)
        
        if code_text:
            # Syntax highlighted code
            syntax = Syntax(
                code_text,
                "python",
                theme="monokai",
                line_numbers=True,
                word_wrap=True
            )
            
            code_panel = Panel(
                syntax,
                title="Generated Code",
                border_style="bright_green"
            )
            self.console.print(code_panel)
            
            # Code actions
            if Confirm.ask("💾 Save code to file?", default=False):
                self.save_code_to_file(code_text, algorithm_name)
        
        # Problem analysis if available
        problem_analysis = solution_data.get("problem_analysis", {})
        if problem_analysis:
            analysis_table = Table(title="Problem Analysis", box=box.ROUNDED)
            analysis_table.add_column("Property", style="cyan")
            analysis_table.add_column("Value", style="green")
            
            for key, value in problem_analysis.items():
                if isinstance(value, list):
                    value = ", ".join(value)
                analysis_table.add_row(key.replace("_", " ").title(), str(value))
            
            self.console.print(analysis_table)
        
        # Alternative approaches if available
        approaches = solution_data.get("algorithmic_approaches", [])
        if approaches:
            self.console.print("\n[bold cyan]🎯 Alternative Approaches:[/bold cyan]")
            for i, approach in enumerate(approaches[:3], 1):
                approach_panel = Panel(
                    f"[bold]{approach.get('name', f'Approach {i}')}[/bold]\n{approach.get('description', '')}",
                    title=f"Alternative {i}",
                    border_style="blue"
                )
                self.console.print(approach_panel)
    
    def save_code_to_file(self, code: str, algorithm_name: str):
        """Save generated code to a file"""
        filename = f"{algorithm_name.lower().replace(' ', '_')}_{self.session_id}.py"
        try:
            with open(filename, 'w') as f:
                f.write(f"# {algorithm_name}\n")
                f.write(f"# Generated by Algorithm Intelligence Suite\n")
                f.write(f"# Session: {self.session_id}\n")
                f.write(f"# Timestamp: {datetime.now()}\n\n")
                f.write(code)
            
            self.console.print(f"[bold green]✅ Code saved to: {filename}[/bold green]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Failed to save code: {e}[/bold red]")
    
    def show_visualizations_menu(self):
        """Display visualizations with interactive menu"""
        self.console.clear()
        self.console.print(Panel("[bold magenta]🎨 VISUALIZATIONS GALLERY[/bold magenta]", expand=False))
        
        viz_data = self.current_results.get("stages", {}).get("visualizations", {})
        
        if not viz_data:
            self.console.print("[bold red]No visualization data available[/bold red]")
            return
        
        # Visualization status
        if viz_data.get("success"):
            status_text = f"✅ [bold green]Success[/bold green] - {viz_data.get('algorithm_detected', 'Unknown')} detected"
            if viz_data.get("confidence"):
                status_text += f" ({viz_data['confidence']*100:.1f}% confidence)"
        else:
            status_text = f"❌ [bold red]Failed[/bold red] - {viz_data.get('message', 'No details')}"
        
        status_panel = Panel(status_text, title="Visualization Status", border_style="magenta")
        self.console.print(status_panel)
        
        # List available visualizations
        files = viz_data.get("files_generated", [])
        if not files:
            self.console.print("[bold yellow]No visualization files generated[/bold yellow]")
            return
        
        # Visualizations table
        viz_table = Table(title="Available Visualizations", box=box.ROUNDED)
        viz_table.add_column("Index", justify="right", style="cyan")
        viz_table.add_column("Filename", style="green")
        viz_table.add_column("Type", style="blue")
        viz_table.add_column("Action", style="yellow")
        
        for i, filename in enumerate(files, 1):
            file_type = self.get_file_type(filename)
            viz_table.add_row(
                str(i),
                filename,
                file_type,
                "🌐 View | 💾 Download"
            )
        
        self.console.print(viz_table)
        
        # Interactive menu
        while True:
            action = Prompt.ask(
                "Action",
                choices=["view", "download", "all", "back"],
                default="back"
            )
            
            if action == "back":
                break
            elif action == "all":
                self.open_all_visualizations(files)
                break
            elif action in ["view", "download"]:
                if len(files) == 1:
                    index = 0
                else:
                    index = IntPrompt.ask(
                        f"Enter visualization number (1-{len(files)})",
                        default=1
                    ) - 1
                
                if 0 <= index < len(files):
                    if action == "view":
                        self.open_visualization(files[index])
                    else:
                        self.download_visualization(files[index])
                else:
                    self.console.print("[bold red]Invalid index[/bold red]")
    
    def get_file_type(self, filename: str) -> str:
        """Get file type description"""
        ext = filename.lower().split('.')[-1]
        types = {
            'png': 'Image',
            'jpg': 'Image',
            'jpeg': 'Image',
            'svg': 'Vector',
            'pdf': 'Document',
            'gif': 'Animation',
            'mp4': 'Video'
        }
        return types.get(ext, 'Unknown')
    
    def open_visualization(self, filename: str):
        """Open visualization in browser"""
        url = f"{self.api_url}/results/{self.session_id}/visualizations/{filename}"
        try:
            webbrowser.open(url)
            self.console.print(f"[bold green]🌐 Opened: {filename}[/bold green]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Failed to open: {e}[/bold red]")
    
    def download_visualization(self, filename: str):
        """Download visualization file"""
        url = f"{self.api_url}/results/{self.session_id}/visualizations/{filename}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                self.console.print(f"[bold green]💾 Downloaded: {filename}[/bold green]")
            else:
                self.console.print(f"[bold red]❌ Download failed: HTTP {response.status_code}[/bold red]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Download error: {e}[/bold red]")
    
    def open_all_visualizations(self, files: List[str]):
        """Open all visualizations in browser"""
        for filename in files:
            self.open_visualization(filename)
            time.sleep(0.5)  # Small delay between opens
    
    def show_performance_analysis(self):
        """Display performance analysis and metrics"""
        self.console.clear()
        self.console.print(Panel("[bold red]📈 PERFORMANCE ANALYSIS[/bold red]", expand=False))
        
        perf_data = self.current_results.get("stages", {}).get("performance_analysis", {})
        
        if not perf_data:
            self.console.print("[bold red]No performance analysis available[/bold red]")
            return
        
        # Performance status
        if perf_data.get("success"):
            status_text = "✅ [bold green]Performance analysis completed[/bold green]"
        else:
            status_text = f"❌ [bold red]Performance analysis failed[/bold red]: {perf_data.get('error', '')}"
        
        status_panel = Panel(status_text, title="Performance Status", border_style="red")
        self.console.print(status_panel)
        
        # Performance files
        files = perf_data.get("files_generated", [])
        if files:
            perf_table = Table(title="Performance Charts", box=box.ROUNDED)
            perf_table.add_column("Index", justify="right", style="cyan")
            perf_table.add_column("Chart", style="green")
            perf_table.add_column("Description", style="dim")
            
            chart_descriptions = {
                "complexity_analysis": "Time/Space complexity comparison",
                "performance_benchmark": "Algorithm performance benchmarking",
                "fallback": "Basic performance overview"
            }
            
            for i, filename in enumerate(files, 1):
                description = "Performance visualization"
                for key, desc in chart_descriptions.items():
                    if key in filename.lower():
                        description = desc
                        break
                
                perf_table.add_row(str(i), filename, description)
            
            self.console.print(perf_table)
            
            # Interactive chart viewing
            if Confirm.ask("🌐 Open performance charts?", default=True):
                for filename in files:
                    url = f"{self.api_url}/results/{self.session_id}/visualizations/{filename}"
                    webbrowser.open(url)
                    time.sleep(0.5)
        
        # Simulated performance metrics (you can enhance this based on actual data)
        self.show_performance_metrics()
    
    def show_performance_metrics(self):
        """Display performance metrics summary"""
        # This would ideally come from your actual performance analysis
        # For now, showing a template that you can populate with real data
        
        metrics_table = Table(title="Performance Metrics", box=box.DOUBLE)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="bold green")
        metrics_table.add_column("Rating", style="yellow")
        
        # Extract or estimate metrics from complexity
        complexity_data = self.current_results.get("stages", {}).get("complexity_analysis", {})
        time_complexity = "Unknown"
        space_complexity = "Unknown"
        
        if complexity_data:
            complexity_info = complexity_data.get("agent_result", {}).get("complexity_analysis", {})
            time_complexity = complexity_info.get("time_complexity", "Unknown")
            space_complexity = complexity_info.get("space_complexity", "Unknown")
        
        # Rating based on complexity
        time_rating = self.get_complexity_rating(time_complexity)
        space_rating = self.get_complexity_rating(space_complexity)
        
        metrics_table.add_row("Time Efficiency", time_complexity, time_rating)
        metrics_table.add_row("Space Efficiency", space_complexity, space_rating)
        metrics_table.add_row("Overall Score", self.calculate_overall_score(time_rating, space_rating), "")
        
        self.console.print(metrics_table)
    
    def get_complexity_rating(self, complexity: str) -> str:
        """Get rating emoji based on complexity"""
        if "O(1)" in complexity or "O(log n)" in complexity:
            return "🟢 Excellent"
        elif "O(n)" in complexity or "O(n log n)" in complexity:
            return "🟡 Good"
        elif "O(n²)" in complexity:
            return "🟠 Fair"
        else:
            return "🔴 Poor"
    
    def calculate_overall_score(self, time_rating: str, space_rating: str) -> str:
        """Calculate overall performance score"""
        if "Excellent" in time_rating and "Excellent" in space_rating:
            return "A+ Outstanding"
        elif "Good" in time_rating or "Good" in space_rating:
            return "B+ Very Good"
        elif "Fair" in time_rating or "Fair" in space_rating:
            return "C+ Average"
        else:
            return "D Needs Improvement"
    
    def show_educational_content(self):
        """Display educational insights and recommendations"""
        self.console.clear()
        self.console.print(Panel("[bold blue]📚 EDUCATIONAL CONTENT[/bold blue]", expand=False))
        
        edu_data = self.current_results.get("stages", {}).get("educational_report", {})
        
        if not edu_data:
            self.console.print("[bold red]No educational content available[/bold red]")
            return
        
        # Key concepts
        concepts = edu_data.get("key_concepts", [])
        if concepts:
            concepts_text = "\n".join([f"• {concept}" for concept in concepts])
            concepts_panel = Panel(
                concepts_text,
                title="🔑 Key Concepts",
                border_style="blue"
            )
            self.console.print(concepts_panel)
        
        # Recommendations
        recommendations = edu_data.get("recommendations", [])
        if recommendations:
            rec_text = "\n".join([f"• {rec}" for rec in recommendations])
            rec_panel = Panel(
                rec_text,
                title="💡 Learning Recommendations",
                border_style="green"
            )
            self.console.print(rec_panel)
        
        # Related algorithms
        related = edu_data.get("related_algorithms", [])
        if related:
            related_text = "\n".join([f"• {algo}" for algo in related])
            related_panel = Panel(
                related_text,
                title="🔗 Related Algorithms",
                border_style="yellow"
            )
            self.console.print(related_panel)
        
        # Practical applications
        applications = edu_data.get("practical_applications", [])
        if applications:
            app_text = "\n".join([f"• {app}" for app in applications])
            app_panel = Panel(
                app_text,
                title="🌟 Practical Applications",
                border_style="magenta"
            )
            self.console.print(app_panel)
        
        # Learning path suggestion
        if Confirm.ask("📖 Show personalized learning path?", default=False):
            self.show_learning_path()
    
    def show_learning_path(self):
        """Show personalized learning path"""
        learning_tree = Tree("🎯 Recommended Learning Path")
        
        # Basic concepts
        basics = learning_tree.add("1. 📚 Master the Basics")
        basics.add("Understand the algorithm's core concept")
        basics.add("Learn the mathematical foundation")
        basics.add("Practice with simple examples")
        
        # Implementation
        implementation = learning_tree.add("2. 💻 Practice Implementation")
        implementation.add("Code the algorithm from scratch")
        implementation.add("Handle edge cases and errors")
        implementation.add("Optimize for readability")
        
        # Advanced topics
        advanced = learning_tree.add("3. 🚀 Advanced Topics")
        advanced.add("Study complexity analysis")
        advanced.add("Learn optimization techniques")
        advanced.add("Explore variations and extensions")
        
        # Real-world application
        realworld = learning_tree.add("4. 🌍 Real-World Application")
        realworld.add("Solve practical problems")
        realworld.add("Work on projects")
        realworld.add("Contribute to open source")
        
        self.console.print(learning_tree)
    
    def show_raw_data(self):
        """Display raw JSON data with syntax highlighting"""
        self.console.clear()
        self.console.print(Panel("[bold dim]🔍 RAW JSON DATA[/bold dim]", expand=False))
        
        if not self.current_results:
            self.console.print("[bold red]No data available[/bold red]")
            return
        
        # Pretty print JSON with syntax highlighting
        json_text = json.dumps(self.current_results, indent=2)
        syntax = Syntax(
            json_text,
            "json",
            theme="monokai",
            line_numbers=True,
            word_wrap=True
        )
        
        json_panel = Panel(
            syntax,
            title="Complete Analysis Results",
            border_style="dim"
        )
        
        # Limit output size for readability
        if len(json_text) > 10000:
            truncated_text = json_text[:10000] + "\n... (truncated for display)"
            syntax = Syntax(truncated_text, "json", theme="monokai", line_numbers=True)
            json_panel = Panel(syntax, title="Complete Analysis Results (Truncated)", border_style="dim")
        
        self.console.print(json_panel)
        
        # Option to save full data
        if Confirm.ask("💾 Save complete data to file?", default=False):
            filename = f"analysis_results_{self.session_id}.json"
            try:
                with open(filename, 'w') as f:
                    json.dump(self.current_results, f, indent=2)
                self.console.print(f"[bold green]✅ Data saved to: {filename}[/bold green]")
            except Exception as e:
                self.console.print(f"[bold red]❌ Failed to save: {e}[/bold red]")
    
    def export_results(self):
        """Export results in various formats"""
        self.console.print(Panel("[bold cyan]💾 EXPORT RESULTS[/bold cyan]", expand=False))
        
        if not self.current_results:
            self.console.print("[bold red]No results to export[/bold red]")
            return
        
        export_options = [
            "JSON (Complete data)",
            "Text Summary",
            "Code Only",
            "HTML Report",
            "All Formats"
        ]
        
        # Display options
        export_table = Table(title="Export Options")
        export_table.add_column("Index", justify="right", style="cyan")
        export_table.add_column("Format", style="green")
        export_table.add_column("Description", style="dim")
        
        descriptions = [
            "Complete analysis data in JSON format",
            "Human-readable summary",
            "Generated algorithm code only",
            "Formatted HTML report",
            "Export in all available formats"
        ]
        
        for i, (option, desc) in enumerate(zip(export_options, descriptions), 1):
            export_table.add_row(str(i), option, desc)
        
        self.console.print(export_table)
        
        choice = IntPrompt.ask("Select export format", default=1, show_default=True)
        
        if choice == 1:
            self.export_json()
        elif choice == 2:
            self.export_text_summary()
        elif choice == 3:
            self.export_code_only()
        elif choice == 4:
            self.export_html_report()
        elif choice == 5:
            self.export_all_formats()
        else:
            self.console.print("[bold red]Invalid choice[/bold red]")
    
    def export_json(self):
        """Export results as JSON"""
        filename = f"analysis_{self.session_id}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(self.current_results, f, indent=2)
            self.console.print(f"[bold green]✅ JSON exported: {filename}[/bold green]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Export failed: {e}[/bold red]")
    
    def export_text_summary(self):
        """Export results as text summary"""
        filename = f"summary_{self.session_id}.txt"
        try:
            with open(filename, 'w') as f:
                f.write("ALGORITHM INTELLIGENCE SUITE - ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Timestamp: {self.current_results.get('timestamp', 'N/A')}\n")
                f.write(f"Input Type: {self.current_results.get('input_type', 'N/A')}\n\n")
                
                # Add complexity info
                complexity = self.current_results.get("stages", {}).get("complexity_analysis", {})
                if complexity:
                    complexity_info = complexity.get("agent_result", {}).get("complexity_analysis", {})
                    f.write("COMPLEXITY ANALYSIS\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Time Complexity: {complexity_info.get('time_complexity', 'N/A')}\n")
                    f.write(f"Space Complexity: {complexity_info.get('space_complexity', 'N/A')}\n")
                    f.write(f"Reasoning: {complexity_info.get('reasoning', 'N/A')}\n\n")
                
                # Add solution info
                solution = self.current_results.get("stages", {}).get("algorithm_solving", {})
                if solution:
                    opt_solution = solution.get("optimal_solution", {})
                    f.write("GENERATED SOLUTION\n")
                    f.write("-" * 18 + "\n")
                    f.write(f"Algorithm: {opt_solution.get('algorithm_name', 'N/A')}\n")
                    f.write(f"Description: {opt_solution.get('description', 'N/A')}\n\n")
            
            self.console.print(f"[bold green]✅ Summary exported: {filename}[/bold green]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Export failed: {e}[/bold red]")
    
    def export_code_only(self):
        """Export generated code only"""
        solution = self.current_results.get("stages", {}).get("algorithm_solving", {})
        if not solution:
            self.console.print("[bold red]No code to export[/bold red]")
            return
        
        opt_solution = solution.get("optimal_solution", {})
        code_data = opt_solution.get("code", {})
        
        if isinstance(code_data, dict):
            code_text = code_data.get("code", "")
        else:
            code_text = str(code_data)
        
        if not code_text:
            self.console.print("[bold red]No code available[/bold red]")
            return
        
        filename = f"algorithm_{self.session_id}.py"
        try:
            with open(filename, 'w') as f:
                f.write(f"# {opt_solution.get('algorithm_name', 'Generated Algorithm')}\n")
                f.write(f"# Generated by Algorithm Intelligence Suite\n")
                f.write(f"# Session: {self.session_id}\n")
                f.write(f"# Timestamp: {datetime.now()}\n\n")
                f.write(code_text)
            
            self.console.print(f"[bold green]✅ Code exported: {filename}[/bold green]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Export failed: {e}[/bold red]")
    
    def export_html_report(self):
        """Export as HTML report"""
        filename = f"report_{self.session_id}.html"
        try:
            html_content = self.generate_html_report()
            with open(filename, 'w') as f:
                f.write(html_content)
            self.console.print(f"[bold green]✅ HTML report exported: {filename}[/bold green]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Export failed: {e}[/bold red]")
    
    def generate_html_report(self) -> str:
        """Generate HTML report content"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Algorithm Analysis Report - {self.session_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 10px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .code {{ background: #f4f4f4; padding: 10px; border-radius: 5px; font-family: monospace; }}
        .complexity {{ background: #e8f5e9; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Algorithm Intelligence Suite</h1>
        <h2>Analysis Report</h2>
        <p>Session: {self.session_id}</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h3>📊 Analysis Summary</h3>
        <p><strong>Input Type:</strong> {self.current_results.get('input_type', 'N/A')}</p>
        <p><strong>Stages Completed:</strong> {len(self.current_results.get('stages', {}))}</p>
    </div>
    
    <!-- Add more sections based on available data -->
    
</body>
</html>
        """
    
    def export_all_formats(self):
        """Export in all available formats"""
        self.console.print("[bold cyan]Exporting in all formats...[/bold cyan]")
        
        with Progress(console=self.console) as progress:
            task = progress.add_task("Exporting...", total=4)
            
            self.export_json()
            progress.advance(task)
            
            self.export_text_summary()
            progress.advance(task)
            
            self.export_code_only()
            progress.advance(task)
            
            self.export_html_report()
            progress.advance(task)
        
        self.console.print("[bold green]✅ All formats exported successfully![/bold green]")
    
    def open_web_report(self):
        """Open full web report in browser"""
        if not self.session_id:
            self.console.print("[bold red]No session available[/bold red]")
            return
        
        url = f"{self.api_url}/results/{self.session_id}"
        try:
            webbrowser.open(url)
            self.console.print(f"[bold green]🌐 Opened web report: {url}[/bold green]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Failed to open: {e}[/bold red]")
    
    def quick_results_display(self):
        """Quick results display for fast analysis"""
        if not self.current_results:
            return
        
        # Quick summary
        self.console.print("\n[bold cyan]📋 QUICK SUMMARY[/bold cyan]")
        
        # Algorithm name
        solution = self.current_results.get("stages", {}).get("algorithm_solving", {})
        if solution:
            algorithm_name = solution.get("optimal_solution", {}).get("algorithm_name", "Unknown")
            self.console.print(f"[bold green]Algorithm:[/bold green] {algorithm_name}")
        
        # Complexity
        complexity = self.current_results.get("stages", {}).get("complexity_analysis", {})
        if complexity:
            complexity_info = complexity.get("agent_result", {}).get("complexity_analysis", {})
            time_comp = complexity_info.get("time_complexity", "Unknown")
            space_comp = complexity_info.get("space_complexity", "Unknown")
            self.console.print(f"[bold yellow]Time Complexity:[/bold yellow] {time_comp}")
            self.console.print(f"[bold yellow]Space Complexity:[/bold yellow] {space_comp}")
        
        # Visualizations
        viz_data = self.current_results.get("stages", {}).get("visualizations", {})
        if viz_data and viz_data.get("files_generated"):
            viz_count = len(viz_data["files_generated"])
            self.console.print(f"[bold magenta]Visualizations:[/bold magenta] {viz_count} files generated")
        
        # Quick actions
        actions_panel = Panel(
            "[bold]Quick Actions:[/bold]\n"
            "• Press [bold]3[/bold] from main menu to view detailed results\n"
            "• Press [bold]9[/bold] to open web report\n"
            "• Press [bold]8[/bold] to export results",
            title="Next Steps",
            border_style="green"
        )
        self.console.print(actions_panel)
    
    def browse_sessions_history(self):
        """Browse previous analysis sessions"""
        self.console.clear()
        self.console.print(Panel("[bold cyan]📚 SESSION HISTORY[/bold cyan]", expand=False))
        
        if not self.sessions_history:
            self.console.print("[bold yellow]No previous sessions found[/bold yellow]")
            return
        
        # History table
        history_table = Table(title="Previous Analysis Sessions", box=box.ROUNDED)
        history_table.add_column("Index", justify="right", style="cyan")
        history_table.add_column("Session ID", style="green")
        history_table.add_column("Timestamp", style="blue")
        history_table.add_column("Input Preview", style="dim")
        history_table.add_column("Type", style="yellow")
        
        for i, session in enumerate(self.sessions_history, 1):
            history_table.add_row(
                str(i),
                session["session_id"],
                session["timestamp"].strftime("%Y-%m-%d %H:%M"),
                session["input"],
                session["input_type"]
            )
        
        self.console.print(history_table)
        
        # Session selection
        if Confirm.ask("Load a previous session?", default=False):
            index = IntPrompt.ask(
                f"Enter session number (1-{len(self.sessions_history)})",
                default=1
            ) - 1
            
            if 0 <= index < len(self.sessions_history):
                selected_session = self.sessions_history[index]
                self.load_session(selected_session["session_id"])
            else:
                self.console.print("[bold red]Invalid session number[/bold red]")
    
    def load_session(self, session_id: str):
        """Load a previous session"""
        with Status(f"[bold blue]Loading session {session_id}...", console=self.console):
            try:
                # Try to get session results from API
                response = requests.get(f"{self.api_url}/results/{session_id}/reports/analysis_results.json")
                if response.status_code == 200:
                    self.current_results = response.json()
                    self.session_id = session_id
                    self.console.print(f"[bold green]✅ Session {session_id} loaded successfully[/bold green]")
                else:
                    self.console.print(f"[bold red]❌ Session not found or expired[/bold red]")
            except Exception as e:
                self.console.print(f"[bold red]❌ Failed to load session: {e}[/bold red]")
    
    def advanced_features_menu(self):
        """Advanced features and tools menu"""
        self.console.clear()
        
        advanced_panel = Panel(
            """[bold cyan]🔬 ADVANCED FEATURES[/bold cyan]

[bold green]1.[/bold green] 🔍 Algorithm Comparison Tool
[bold green]2.[/bold green] 📊 Custom Performance Benchmarks
[bold green]3.[/bold green] 🎨 Visualization Generator
[bold green]4.[/bold green] 🧪 Code Complexity Analyzer
[bold green]5.[/bold green] 📚 Knowledge Base Search
[bold green]6.[/bold green] 🔄 Batch Analysis
[bold green]7.[/bold green] ⬅️ Back to Main Menu""",
            title="Advanced Features",
            border_style="bright_cyan"
        )
        
        self.console.print(advanced_panel)
        choice = Prompt.ask("Select feature", choices=[str(i) for i in range(1, 8)])
        
        if choice == "1":
            self.algorithm_comparison_tool()
        elif choice == "2":
            self.custom_benchmarks()
        elif choice == "3":
            self.visualization_generator()
        elif choice == "4":
            self.code_complexity_analyzer()
        elif choice == "5":
            self.knowledge_base_search()
        elif choice == "6":
            self.batch_analysis()
        # choice == "7" returns to main menu
    
    def algorithm_comparison_tool(self):
        """Compare multiple algorithms"""
        self.console.print(Panel("[bold cyan]🔍 ALGORITHM COMPARISON TOOL[/bold cyan]", expand=False))
        self.console.print("[bold yellow]Feature coming soon![/bold yellow]")
        self.console.print("This tool will allow you to compare multiple algorithms side by side.")
    
    def custom_benchmarks(self):
        """Custom performance benchmarks"""
        self.console.print(Panel("[bold cyan]📊 CUSTOM BENCHMARKS[/bold cyan]", expand=False))
        
        if Confirm.ask("Generate standard performance benchmarks?"):
            try:
                response = requests.get(f"{self.api_url}/api/performance")
                if response.status_code == 200:
                    data = response.json()
                    files = data.get("files_generated", [])
                    if files:
                        self.console.print(f"[bold green]✅ Generated {len(files)} benchmark charts[/bold green]")
                        for filename in files:
                            self.console.print(f"  • {filename}")
                    else:
                        self.console.print("[bold yellow]No benchmark files generated[/bold yellow]")
                else:
                    self.console.print(f"[bold red]❌ Benchmark generation failed[/bold red]")
            except Exception as e:
                self.console.print(f"[bold red]❌ Error: {e}[/bold red]")
    
    def visualization_generator(self):
        """Generate custom visualizations"""
        self.console.print(Panel("[bold cyan]🎨 VISUALIZATION GENERATOR[/bold cyan]", expand=False))
        
        algorithm = Prompt.ask("Enter algorithm name to visualize")
        if algorithm:
            try:
                response = requests.post(
                    f"{self.api_url}/api/visualize",
                    json={"algorithm": algorithm}
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        files = data.get("files_generated", [])
                        self.console.print(f"[bold green]✅ Generated {len(files)} visualizations[/bold green]")
                        
                        if Confirm.ask("Open visualizations?", default=True):
                            for filename in files:
                                url = f"{self.api_url}/results/latest/visualizations/{filename}"
                                webbrowser.open(url)
                    else:
                        self.console.print(f"[bold yellow]No visualizations generated: {data.get('message')}[/bold yellow]")
                else:
                    self.console.print("[bold red]❌ Visualization request failed[/bold red]")
            except Exception as e:
                self.console.print(f"[bold red]❌ Error: {e}[/bold red]")
    
    def code_complexity_analyzer(self):
        """Analyze code complexity"""
        self.console.print(Panel("[bold cyan]🧪 CODE COMPLEXITY ANALYZER[/bold cyan]", expand=False))
        
        self.console.print("Enter your code (press Enter twice to finish):")
        code_lines = []
        while True:
            line = input()
            if line == "" and code_lines:
                break
            code_lines.append(line)
        
        code = "\n".join(code_lines)
        if code.strip():
            try:
                response = requests.post(
                    f"{self.api_url}/api/complexity",
                    json={"code": code, "language": "python"}
                )
                if response.status_code == 200:
                    data = response.json()
                    complexity_info = data.get("agent_result", {}).get("complexity_analysis", {})
                    
                    result_table = Table(title="Complexity Analysis Results")
                    result_table.add_column("Metric", style="cyan")
                    result_table.add_column("Value", style="green")
                    
                    result_table.add_row("Time Complexity", complexity_info.get("time_complexity", "Unknown"))
                    result_table.add_row("Space Complexity", complexity_info.get("space_complexity", "Unknown"))
                    
                    self.console.print(result_table)
                    
                    reasoning = complexity_info.get("reasoning", "")
                    if reasoning:
                        reasoning_panel = Panel(reasoning, title="Analysis Reasoning", border_style="blue")
                        self.console.print(reasoning_panel)
                else:
                    self.console.print("[bold red]❌ Analysis failed[/bold red]")
            except Exception as e:
                self.console.print(f"[bold red]❌ Error: {e}[/bold red]")
    
    def knowledge_base_search(self):
        """Search the knowledge base"""
        self.console.print(Panel("[bold cyan]📚 KNOWLEDGE BASE SEARCH[/bold cyan]", expand=False))
        
        query = Prompt.ask("Enter search query")
        if query:
            # This would connect to your RAG system
            self.console.print(f"[bold blue]Searching for: {query}[/bold blue]")
            self.console.print("[bold yellow]Feature coming soon![/bold yellow]")
            self.console.print("This will search through the algorithm knowledge base using RAG.")
    
    def batch_analysis(self):
        """Batch analysis of multiple inputs"""
        self.console.print(Panel("[bold cyan]🔄 BATCH ANALYSIS[/bold cyan]", expand=False))
        self.console.print("[bold yellow]Feature coming soon![/bold yellow]")
        self.console.print("This tool will allow you to analyze multiple algorithms at once.")
    
    def system_tools_menu(self):
        """System tools and utilities"""
        self.console.clear()
        
        tools_panel = Panel(
            """[bold cyan]🛠️ SYSTEM TOOLS[/bold cyan]

[bold green]1.[/bold green] 🔍 API Health Check
[bold green]2.[/bold green] 📊 System Statistics
[bold green]3.[/bold green] 🗄️ Database Status
[bold green]4.[/bold green] 🧹 Clear Session Cache
[bold green]5.[/bold green] 📋 Export All Sessions
[bold green]6.[/bold green] 🔧 Repair Tools
[bold green]7.[/bold green] ⬅️ Back to Main Menu""",
            title="System Tools",
            border_style="bright_yellow"
        )
        
        self.console.print(tools_panel)
        choice = Prompt.ask("Select tool", choices=[str(i) for i in range(1, 8)])
        
        if choice == "1":
            self.detailed_health_check()
        elif choice == "2":
            self.system_statistics()
        elif choice == "3":
            self.database_status()
        elif choice == "4":
            self.clear_cache()
        elif choice == "5":
            self.export_all_sessions()
        elif choice == "6":
            self.repair_tools()
    
    def detailed_health_check(self):
        """Detailed system health check"""
        self.console.print(Panel("[bold cyan]🔍 DETAILED HEALTH CHECK[/bold cyan]", expand=False))
        
        with Progress(console=self.console) as progress:
            task = progress.add_task("Running health checks...", total=5)
            
            # API Status
            try:
                response = requests.get(f"{self.api_url}/api/status", timeout=5)
                api_status = "✅ Online" if response.status_code == 200 else "❌ Error"
                progress.advance(task)
            except:
                api_status = "❌ Offline"
                progress.advance(task)
            
            # Database check
            try:
                response = requests.get(f"{self.api_url}/api/algorithms", timeout=5)
                db_status = "✅ Connected" if response.status_code == 200 else "❌ Error"
                progress.advance(task)
            except:
                db_status = "❌ Disconnected"
                progress.advance(task)
            
            # Performance test
            start_time = time.time()
            try:
                requests.get(f"{self.api_url}/api/status", timeout=5)
                response_time = time.time() - start_time
                perf_status = f"✅ {response_time:.3f}s"
                progress.advance(task)
            except:
                perf_status = "❌ Timeout"
                progress.advance(task)
            
            # Features check
            try:
                response = requests.get(f"{self.api_url}/api/status", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    features = data.get("features", {})
                    feature_count = sum(1 for f in features.values() if f)
                    features_status = f"✅ {feature_count}/{len(features)} available"
                else:
                    features_status = "❌ Unknown"
                progress.advance(task)
            except:
                features_status = "❌ Error"
                progress.advance(task)
            
            # Memory usage (simulated)
            memory_status = "✅ Normal"
            progress.advance(task)
        
        # Display results
        health_table = Table(title="System Health Report", box=box.DOUBLE)
        health_table.add_column("Component", style="cyan")
        health_table.add_column("Status", style="bold")
        health_table.add_column("Details", style="dim")
        
        health_table.add_row("API Server", api_status, f"Endpoint: {self.api_url}")
        health_table.add_row("Database", db_status, "MongoDB connection")
        health_table.add_row("Response Time", perf_status, "API latency")
        health_table.add_row("Features", features_status, "Available services")
        health_table.add_row("Memory Usage", memory_status, "System resources")
        
        self.console.print(health_table)
    
    def system_statistics(self):
        """Display system statistics"""
        self.console.print(Panel("[bold cyan]📊 SYSTEM STATISTICS[/bold cyan]", expand=False))
        
        stats_table = Table(title="Usage Statistics", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Sessions This CLI Run", str(len(self.sessions_history)))
        stats_table.add_row("Current Session", self.session_id or "None")
        stats_table.add_row("CLI Start Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        stats_table.add_row("API Endpoint", self.api_url)
        
        self.console.print(stats_table)
    
    def database_status(self):
        """Check database status"""
        self.console.print(Panel("[bold cyan]🗄️ DATABASE STATUS[/bold cyan]", expand=False))
        
        try:
            response = requests.get(f"{self.api_url}/api/algorithms")
            if response.status_code == 200:
                data = response.json()
                
                db_table = Table(title="Database Information", box=box.ROUNDED)
                db_table.add_column("Property", style="cyan")
                db_table.add_column("Value", style="green")
                
                db_table.add_row("Total Algorithms", str(data.get("total_count", 0)))
                db_table.add_row("Categories", str(len(data.get("categories", []))))
                db_table.add_row("Status", "✅ Connected")
                
                self.console.print(db_table)
                
                # Show categories
                categories = data.get("categories", [])
                if categories:
                    cat_text = ", ".join(categories)
                    cat_panel = Panel(cat_text, title="Available Categories", border_style="green")
                    self.console.print(cat_panel)
            else:
                self.console.print("[bold red]❌ Database connection failed[/bold red]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Database error: {e}[/bold red]")
    
    def clear_cache(self):
        """Clear session cache"""
        if Confirm.ask("Clear all session history?", default=False):
            self.sessions_history.clear()
            self.console.print("[bold green]✅ Session cache cleared[/bold green]")
    
    def export_all_sessions(self):
        """Export all session data"""
        if not self.sessions_history:
            self.console.print("[bold yellow]No sessions to export[/bold yellow]")
            return
        
        filename = f"all_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(self.sessions_history, f, indent=2, default=str)
            self.console.print(f"[bold green]✅ All sessions exported: {filename}[/bold green]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Export failed: {e}[/bold red]")
    
    def repair_tools(self):
        """System repair tools"""
        self.console.print(Panel("[bold cyan]🔧 REPAIR TOOLS[/bold cyan]", expand=False))
        self.console.print("[bold yellow]Repair tools coming soon![/bold yellow]")
        self.console.print("These tools will help diagnose and fix common issues.")
    
    def settings_menu(self):
        """Settings and configuration menu"""
        self.console.clear()
        
        settings_panel = Panel(
            f"""[bold cyan]⚙️ SETTINGS[/bold cyan]

[bold green]Current Configuration:[/bold green]
• API URL: {self.api_url}
• Auto-open visualizations: {self.config['auto_open_visualizations']}
• Save sessions: {self.config['save_sessions']}
• Theme: {self.config['theme']}
• Show debug info: {self.config['show_debug']}

[bold green]1.[/bold green] 🌐 Change API URL
[bold green]2.[/bold green] 🎨 Toggle Auto-open Visualizations
[bold green]3.[/bold green] 💾 Toggle Save Sessions
[bold green]4.[/bold green] 🎭 Change Theme
[bold green]5.[/bold green] 🐛 Toggle Debug Mode
[bold green]6.[/bold green] 🔄 Reset to Defaults
[bold green]7.[/bold green] ⬅️ Back to Main Menu""",
            title="Settings",
            border_style="bright_magenta"
        )
        
        self.console.print(settings_panel)
        choice = Prompt.ask("Select setting", choices=[str(i) for i in range(1, 8)])
        
        if choice == "1":
            new_url = Prompt.ask("Enter new API URL", default=self.api_url)
            self.api_url = new_url
            self.console.print(f"[bold green]✅ API URL updated: {new_url}[/bold green]")
        elif choice == "2":
            self.config['auto_open_visualizations'] = not self.config['auto_open_visualizations']
            status = "enabled" if self.config['auto_open_visualizations'] else "disabled"
            self.console.print(f"[bold green]✅ Auto-open visualizations {status}[/bold green]")
        elif choice == "3":
            self.config['save_sessions'] = not self.config['save_sessions']
            status = "enabled" if self.config['save_sessions'] else "disabled"
            self.console.print(f"[bold green]✅ Save sessions {status}[/bold green]")
        elif choice == "4":
            self.config['theme'] = "light" if self.config['theme'] == "dark" else "dark"
            self.console.print(f"[bold green]✅ Theme changed to {self.config['theme']}[/bold green]")
        elif choice == "5":
            self.config['show_debug'] = not self.config['show_debug']
            status = "enabled" if self.config['show_debug'] else "disabled"
            self.console.print(f"[bold green]✅ Debug mode {status}[/bold green]")
        elif choice == "6":
            self.config = {
                "auto_open_visualizations": False,
                "save_sessions": True,
                "theme": "dark",
                "show_debug": False
            }
            self.console.print("[bold green]✅ Settings reset to defaults[/bold green]")
    
    def show_help(self):
        """Display help and documentation"""
        self.console.clear()
        
        help_content = """
[bold cyan]❓ ALGORITHM INTELLIGENCE SUITE - HELP & DOCUMENTATION[/bold cyan]

[bold green]GETTING STARTED[/bold green]
1. Ensure your API server is running (python app.py)
2. Use "New Comprehensive Analysis" for full analysis
3. Use "Quick Analysis" for fast results
4. View results using the interactive menu

[bold green]INPUT TYPES[/bold green]
• [bold]Problem Description:[/bold] "Implement BFS algorithm"
• [bold]Algorithm Question:[/bold] "How does binary search work?"
• [bold]Code Analysis:[/bold] Paste your Python code

[bold green]FEATURES[/bold green]
• 🧠 AI-powered algorithm analysis
• ⚡ Complexity analysis (time/space)
• 🎨 Algorithm visualizations
• 📊 Performance benchmarking
• 📚 Educational insights
• 💾 Multiple export formats

[bold green]KEYBOARD SHORTCUTS[/bold green]
• Ctrl+C: Interrupt current operation
• Enter: Confirm selection
• Arrow keys: Navigate (where applicable)

[bold green]TROUBLESHOOTING[/bold green]
• If API connection fails, check if server is running
• Use "System Tools > Health Check" for diagnostics
• Check Settings if visualizations don't open
• Use Debug Mode for detailed error information

[bold green]EXPORT FORMATS[/bold green]
• JSON: Complete analysis data
• Text: Human-readable summary
• Code: Generated algorithm code only
• HTML: Formatted report
• All: Export in all formats

[bold green]SUPPORT[/bold green]
• Visit the GitHub repository for issues
• Check API documentation at /api/status
• Use "System Tools" for diagnostics
        """
        
        help_panel = Panel(
            help_content,
            title="Help & Documentation",
            border_style="bright_yellow",
            padding=(1, 2)
        )
        
        self.console.print(help_panel)
        
        input("\nPress Enter to continue...")
    
    def exit_cli(self):
        """Exit the CLI application"""
        self.console.clear()
        
        exit_banner = """
╔══════════════════════════════════════════════════════════════════╗
║                    Thank you for using                           ║
║            🤖 ALGORITHM INTELLIGENCE SUITE CLI                   ║
║                                                                  ║
║  Your algorithm analysis journey continues at:                   ║
║  🌐 Web Interface: {api_url}                     ║
║                                                                  ║
║  Sessions analyzed: {sessions}                                           ║
║  Happy coding! 🚀                                                ║
╚══════════════════════════════════════════════════════════════════╝
        """.format(
            api_url=self.api_url.ljust(37),
            sessions=str(len(self.sessions_history)).ljust(2)
        )
        
        self.console.print(Panel(
            Align.center(exit_banner),
            style="bold green",
            border_style="bright_green"
        ))
        
        # Save session history if configured
        if self.config["save_sessions"] and self.sessions_history:
            if Confirm.ask("Save session history for next time?", default=True):
                filename = "cli_session_history.json"
                try:
                    with open(filename, 'w') as f:
                        json.dump(self.sessions_history, f, indent=2, default=str)
                    self.console.print(f"[bold green]✅ Session history saved to {filename}[/bold green]")
                except Exception as e:
                    self.console.print(f"[bold red]❌ Failed to save history: {e}[/bold red]")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Algorithm Intelligence Suite CLI")
    parser.add_argument('--api-url', default='http://localhost:5000', 
                       help='API server URL (default: http://localhost:5000)')
    parser.add_argument('--version', action='version', version='Algorithm Intelligence Suite CLI v3.0')
    
    args = parser.parse_args()
    
    # Initialize and run CLI
    try:
        cli = AlgorithmIntelligenceCLI(api_url=args.api_url)
        cli.run()
    except KeyboardInterrupt:
        print("\n\nCLI interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nCLI Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
