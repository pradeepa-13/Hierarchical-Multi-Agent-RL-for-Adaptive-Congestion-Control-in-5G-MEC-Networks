#!/usr/bin/env python3
"""
Enhanced 5G MEC Network Metrics Visualization Tool
Features:
- Interactive subplot viewing (click to enlarge)
- Real-time and post-simulation modes
- Professional styling with better spacing
- Comprehensive network metrics analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import xml.etree.ElementTree as ET
import os
import sys
import time
from typing import Dict
from datetime import datetime

# Configuration
CSV_FILE = "mec_metrics.csv"
XML_FILE = "mec_flowmon.xml"
UPDATE_INTERVAL = 2000  # milliseconds for real-time updates

# Enhanced color palette
COLORS = {
    'eMBB': '#3498db',      # Blue
    'URLLC': '#e74c3c',     # Red
    'mMTC': '#2ecc71',      # Green
    'Background': '#f39c12', # Orange
    'Unknown': '#95a5a6'    # Gray
}

class MetricsVisualizer:
    def __init__(self, mode='realtime'):
        """
        mode: 'realtime' or 'postsim'
        """
        self.mode = mode
        self.last_size = 0
        self.enlarged_plot = None
        self.main_fig = None
        self.enlarged_fig = None
        
        # Setup matplotlib with professional style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 16
        plt.rcParams['figure.titleweight'] = 'bold'
        
        self.setup_main_figure()
        
        # Initialize data storage
        self.data_history = []
    
    def setup_main_figure(self):
        """Create the main dashboard with all subplots"""
        self.main_fig = plt.figure(figsize=(20, 12))
        self.main_fig.suptitle('5G MEC Network Performance Metrics Dashboard\n(Click any plot to enlarge)', 
                               fontsize=18, fontweight='bold', y=0.98)
        
        # Create grid layout with better spacing
        self.gs = GridSpec(3, 3, figure=self.main_fig, 
                          hspace=0.4, wspace=0.35,
                          left=0.06, right=0.96, 
                          top=0.93, bottom=0.05)
        
        # Create subplots
        self.ax1 = self.main_fig.add_subplot(self.gs[0, :2])  # Throughput timeline
        self.ax2 = self.main_fig.add_subplot(self.gs[0, 2])   # Service type pie
        self.ax3 = self.main_fig.add_subplot(self.gs[1, :2])  # Delay over time
        self.ax4 = self.main_fig.add_subplot(self.gs[1, 2])   # Loss rate bar
        self.ax5 = self.main_fig.add_subplot(self.gs[2, 0])   # Delay CDF
        self.ax6 = self.main_fig.add_subplot(self.gs[2, 1])   # Jitter histogram
        self.ax7 = self.main_fig.add_subplot(self.gs[2, 2])   # CWND Evolution (CHANGED!)
        
        # Store subplot info for click handling
        self.subplot_info = {
            self.ax1: {'title': 'Throughput Over Time', 'type': 'throughput'},
            self.ax2: {'title': 'Traffic Distribution', 'type': 'pie'},
            self.ax3: {'title': 'Average Delay Over Time', 'type': 'delay'},
            self.ax4: {'title': 'Packet Loss Rate', 'type': 'loss'},
            self.ax5: {'title': 'Delay CDF', 'type': 'cdf'},
            self.ax6: {'title': 'Jitter Distribution', 'type': 'jitter'},
            self.ax7: {'title': 'TCP CWND Evolution', 'type': 'cwnd'}  # CHANGED!
        }
        
        # Connect click event
        self.main_fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add instruction text at bottom
        self.main_fig.text(0.5, 0.01, 
                          '💡 Click on any subplot to view enlarged | Press ESC to return to dashboard',
                          ha='center', va='bottom', fontsize=11, 
                          style='italic', color='#34495e',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.8))
    
    def on_click(self, event):
        """Handle click events on subplots"""
        if event.inaxes is None:
            return
        
        # Check if click is on one of our subplots
        if event.inaxes in self.subplot_info:
            self.enlarge_plot(event.inaxes)
    
    def on_key_press(self, event):
        """Handle keyboard events in enlarged view"""
        if event.key == 'escape':
            self.return_to_dashboard()
    
    def enlarge_plot(self, ax):
        """Create an enlarged view of the selected subplot"""
        plot_info = self.subplot_info[ax]
        
        # Close existing enlarged figure if any
        if self.enlarged_fig is not None:
            plt.close(self.enlarged_fig)
        
        # Create new enlarged figure
        self.enlarged_fig = plt.figure(figsize=(16, 10))
        self.enlarged_fig.canvas.manager.set_window_title(f'Enlarged View: {plot_info["title"]}')
        
        # Add title and instructions
        self.enlarged_fig.suptitle(f'{plot_info["title"]} - Detailed View', 
                                   fontsize=20, fontweight='bold', y=0.96)
        self.enlarged_fig.text(0.5, 0.02, 
                              'Press ESC to return to dashboard',
                              ha='center', va='bottom', fontsize=12, 
                              style='italic', color='#2c3e50',
                              bbox=dict(boxstyle='round,pad=0.7', 
                                      facecolor='#e8f4f8', alpha=0.9))
        
        # Create enlarged axis with margins
        enlarged_ax = self.enlarged_fig.add_subplot(111)
        self.enlarged_fig.subplots_adjust(left=0.08, right=0.94, top=0.92, bottom=0.08)
        
        # Store reference for updates
        self.enlarged_ax = enlarged_ax
        self.enlarged_plot_type = plot_info['type']
        self.current_enlarged_data = None  # Store data for this plot
        
        # Connect keyboard event
        self.enlarged_fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Load and plot data - force reload for real-time mode
        if self.mode == 'realtime':
            # In realtime mode, force a fresh data load
            old_size = self.last_size
            self.last_size = 0  # Force reload
            df = self.load_csv_data()
            self.last_size = old_size  # Restore
        else:
            df = self.load_csv_data()
            
        if df is not None and not df.empty:
            self.current_enlarged_data = df  # Store for updates
            self.update_enlarged_plot(df, enlarged_ax, plot_info['type'])
        else:
            # Show a message if no data available yet
            enlarged_ax.text(0.5, 0.5, 'Waiting for data...\n(No metrics available yet)', 
                           ha='center', va='center', fontsize=16, color='#7f8c8d',
                           transform=enlarged_ax.transAxes)
            enlarged_ax.set_xticks([])
            enlarged_ax.set_yticks([])
        
        # Hide main figure and show enlarged
        if self.main_fig:
            try:
                self.main_fig.canvas.manager.window.withdraw() if hasattr(self.main_fig.canvas.manager, 'window') else None
            except:
                pass
        
        plt.figure(self.enlarged_fig.number)
        plt.show(block=False)
        plt.pause(0.1)
    
    def return_to_dashboard(self):
        """Return to the main dashboard view"""
        if self.enlarged_fig is not None:
            plt.close(self.enlarged_fig)
            self.enlarged_fig = None
            self.enlarged_ax = None
            self.enlarged_plot_type = None
        
        # Show main figure
        if self.main_fig:
            try:
                self.main_fig.canvas.manager.window.deiconify() if hasattr(self.main_fig.canvas.manager, 'window') else None
                plt.figure(self.main_fig.number)
                plt.show(block=False)
                plt.pause(0.1)
            except:
                pass
    
    def update_enlarged_plot(self, df, ax, plot_type):
        """Update the enlarged plot with data"""
        ax.clear()
        
        if plot_type == 'throughput':
            self.plot_throughput_detailed(df, ax)
        elif plot_type == 'pie':
            self.plot_distribution_detailed(df, ax)
        elif plot_type == 'delay':
            self.plot_delay_detailed(df, ax)
        elif plot_type == 'loss':
            self.plot_loss_detailed(df, ax)
        elif plot_type == 'cdf':
            self.plot_cdf_detailed(df, ax)
        elif plot_type == 'jitter':
            self.plot_jitter_detailed(df, ax)
        elif plot_type == 'flows':
            self.plot_flows_detailed(df, ax)
        elif plot_type == 'cwnd':  # NEW CASE
            self.plot_cwnd_detailed(df, ax)
        
        try:
            self.enlarged_fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.94])
        except:
            pass
    
    def plot_throughput_detailed(self, df, ax):
        """Detailed throughput plot with hover tooltips"""
        ax.set_title('Throughput Over Time (by Service Type)', fontsize=16, pad=20)
        ax.set_xlabel('Time (s)', fontsize=14, labelpad=10)
        ax.set_ylabel('Throughput (Mbps)', fontsize=14, labelpad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        lines = []
        for service in ['eMBB', 'URLLC', 'mMTC', 'Background']:
            service_data = df[df['service_type'] == service]
            if not service_data.empty:
                agg_data = service_data.groupby('timestamp')['throughput_mbps'].sum().reset_index()
                line, = ax.plot(agg_data['timestamp'], agg_data['throughput_mbps'], 
                               marker='o', label=service, color=COLORS[service],
                               linewidth=3, markersize=6, alpha=0.8)
                lines.append((line, agg_data, service))
        
        ax.legend(loc='upper left', fontsize=13, framealpha=0.9)
        ax.tick_params(labelsize=12)
        
        # Add hover annotation
        annot = ax.annotate("", xy=(0,0), xytext=(20,20),
                           textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.8", fc="#ecf0f1", ec="#34495e", lw=2, alpha=0.95),
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=2),
                           fontsize=11, weight='bold', color='#2c3e50')
        annot.set_visible(False)
        
        def hover_line(event):
            if event.inaxes == ax:
                for line, data, service in lines:
                    cont, ind = line.contains(event)
                    if cont:
                        idx = ind["ind"][0]
                        x = data['timestamp'].iloc[idx]
                        y = data['throughput_mbps'].iloc[idx]
                        annot.xy = (x, y)
                        text = f"{service}\nTime: {x:.2f}s\nThroughput: {y:.2f} Mbps"
                        annot.set_text(text)
                        annot.set_visible(True)
                        self.enlarged_fig.canvas.draw_idle()
                        return
                annot.set_visible(False)
                self.enlarged_fig.canvas.draw_idle()
        
        if self.enlarged_fig is not None:
            self.enlarged_fig.canvas.mpl_connect("motion_notify_event", hover_line)
    
    def plot_distribution_detailed(self, df, ax):
        """Detailed traffic distribution pie chart with hover tooltips"""
        
        # ✅ CORRECTED: Calculate aggregate throughput per service type
        # Instead of summing per-flow throughputs, we calculate true aggregate
        
        # Get time range for observation window
        timestamps = df['timestamp'].unique()
        if len(timestamps) < 2:
            observation_window = df['timestamp'].max() if len(timestamps) == 1 else 1.0
        else:
            observation_window = timestamps[-1] - timestamps[0]
        
        # Prevent division by zero
        if observation_window < 0.1:
            observation_window = 1.0
        
        # Calculate aggregate throughput per service type
        service_throughput = {}
        
        for service in ['eMBB', 'URLLC', 'mMTC', 'Background', 'Unknown']:
            service_data = df[df['service_type'] == service]
            
            if service_data.empty:
                continue
            
            # Get the latest timestamp data for each flow to get cumulative rx_packets
            latest_time = service_data['timestamp'].max()
            latest_data = service_data[service_data['timestamp'] == latest_time]
            
            # Calculate total received bytes (approximate from packets)
            # Assuming average packet size of 1400 bytes (you can adjust)
            total_rx_packets = latest_data['rx_packets'].sum()
            total_rx_bytes = total_rx_packets * 1400  # Bytes
            
            # Convert to Mbps: (bytes * 8 bits/byte) / (time in seconds * 1,000,000)
            throughput_mbps = (total_rx_bytes * 8.0) / (observation_window * 1000000.0)
            
            service_throughput[service] = throughput_mbps
        
        # Convert to pandas Series for easier handling
        service_bytes = pd.Series(service_throughput)
        
        # Remove services with zero throughput
        service_bytes = service_bytes[service_bytes > 0]
        
        if service_bytes.empty:
            ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', fontsize=16, color='#7f8c8d',
                transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Colors and explode settings
        colors = [COLORS.get(s, COLORS['Unknown']) for s in service_bytes.index]
        explode = [0.08 if s == 'eMBB' else 0.04 for s in service_bytes.index]
        
        # Create pie chart without labels initially
        wedges, texts, autotexts = ax.pie(
            service_bytes.values, 
            labels=None,  # No labels to avoid overlap
            autopct='%1.1f%%',
            colors=colors, 
            explode=explode, 
            startangle=90,
            pctdistance=0.75,
            textprops={'fontsize': 14, 'weight': 'bold'}
        )
        
        # Make percentage text more visible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_weight('bold')
        
        # Add legend instead of labels to avoid overlap
        ax.legend(
            wedges, 
            [f'{label}\n({value:.2f} Mbps)' 
            for label, value in zip(service_bytes.index, service_bytes.values)],
            title="Service Types",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=13,
            title_fontsize=14
        )
        
        # Add total throughput to title
        total_throughput = service_bytes.sum()
        ax.set_title(
            f'Traffic Distribution by Service Type\nTotal: {total_throughput:.2f} Mbps', 
            fontsize=16, 
            pad=20
        )
        
        # Add hover annotation
        annot = ax.annotate(
            "", xy=(0,0), xytext=(20,20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.8", fc="#ecf0f1", ec="#34495e", lw=2, alpha=0.95),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=2),
            fontsize=12, weight='bold', color='#2c3e50'
        )
        annot.set_visible(False)
        
        def hover_pie(event):
            if event.inaxes == ax:
                for i, wedge in enumerate(wedges):
                    if wedge.contains(event)[0]:
                        service = service_bytes.index[i]
                        value = service_bytes.values[i]
                        percentage = (value / service_bytes.sum()) * 100
                        annot.xy = (event.xdata, event.ydata)
                        text = f"{service}\n{value:.2f} Mbps\n({percentage:.1f}%)"
                        annot.set_text(text)
                        annot.set_visible(True)
                        self.enlarged_fig.canvas.draw_idle()
                        return
                annot.set_visible(False)
                self.enlarged_fig.canvas.draw_idle()
        
        if self.enlarged_fig is not None:
            self.enlarged_fig.canvas.mpl_connect("motion_notify_event", hover_pie)
    
    def plot_delay_detailed(self, df, ax):
        """Detailed delay plot with hover tooltips"""
        ax.set_title('Average Delay Over Time (URLLC Critical)', fontsize=16, pad=20)
        ax.set_xlabel('Time (s)', fontsize=14, labelpad=10)
        ax.set_ylabel('Delay (ms)', fontsize=14, labelpad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        # URLLC threshold line
        ax.axhline(y=70, color='#c0392b', linestyle='--', linewidth=3, 
                  label='URLLC Target (<70ms)', alpha=0.7, zorder=1)

        lines = []
        for service in ['eMBB', 'URLLC', 'mMTC']:
            service_data = df[df['service_type'] == service]
            if not service_data.empty:
                agg_delay = service_data.groupby('timestamp')['avg_delay_ms'].mean().reset_index()
                line, = ax.plot(agg_delay['timestamp'], agg_delay['avg_delay_ms'],
                               marker='o', label=service, color=COLORS[service],
                               linewidth=3, markersize=6, alpha=0.8, zorder=2)
                lines.append((line, agg_delay, service))
        
        ax.legend(loc='upper left', fontsize=13, framealpha=0.9)
        ax.tick_params(labelsize=12)
        
        # Add hover annotation
        annot = ax.annotate("", xy=(0,0), xytext=(20,20),
                           textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.8", fc="#ecf0f1", ec="#34495e", lw=2, alpha=0.95),
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=2),
                           fontsize=11, weight='bold', color='#2c3e50')
        annot.set_visible(False)
        
        def hover_line(event):
            if event.inaxes == ax:
                for line, data, service in lines:
                    cont, ind = line.contains(event)
                    if cont:
                        idx = ind["ind"][0]
                        x = data['timestamp'].iloc[idx]
                        y = data['avg_delay_ms'].iloc[idx]
                        annot.xy = (x, y)
                        sla_status = "✓ Within SLA" if y < 0 and service == 'URLLC' else ""
                        text = f"{service}\nTime: {x:.2f}s\nDelay: {y:.2f} ms{' ' + sla_status if sla_status else ''}"
                        annot.set_text(text)
                        annot.set_visible(True)
                        self.enlarged_fig.canvas.draw_idle()
                        return
                annot.set_visible(False)
                self.enlarged_fig.canvas.draw_idle()
        
        if self.enlarged_fig is not None:
            self.enlarged_fig.canvas.mpl_connect("motion_notify_event", hover_line)
    
    def plot_loss_detailed(self, df, ax):
        """Detailed loss rate bar chart with hover tooltips"""
        ax.set_title('Packet Loss Rate by Service Type', fontsize=16, pad=20)
        ax.set_ylabel('Loss Rate (%)', fontsize=14, labelpad=10)
        ax.set_xlabel('Service Type', fontsize=14, labelpad=10)
        
        loss_by_service = df.groupby('service_type')['loss_rate'].mean()
        colors_bars = [COLORS.get(s, COLORS['Unknown']) for s in loss_by_service.index]
        
        bars = ax.bar(range(len(loss_by_service)), loss_by_service.values,
                     color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(range(len(loss_by_service)))
        ax.set_xticklabels(loss_by_service.index, rotation=0, fontsize=13)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.tick_params(labelsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%', ha='center', va='bottom', 
                   fontsize=13, fontweight='bold')
        
        # Add hover annotation
        annot = ax.annotate("", xy=(0,0), xytext=(20,20),
                           textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.8", fc="#ecf0f1", ec="#34495e", lw=2, alpha=0.95),
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=2),
                           fontsize=11, weight='bold', color='#2c3e50')
        annot.set_visible(False)
        
        def hover_bar(event):
            if event.inaxes == ax:
                for i, bar in enumerate(bars):
                    if bar.contains(event)[0]:
                        service = loss_by_service.index[i]
                        loss = loss_by_service.values[i]
                        service_data = df[df['service_type'] == service]
                        tx = service_data['tx_packets'].sum()
                        rx = service_data['rx_packets'].sum()
                        lost = service_data['lost_packets'].sum()
                        annot.xy = (bar.get_x() + bar.get_width()/2, bar.get_height())
                        text = f"{service}\nLoss Rate: {loss:.2f}%\nTX: {tx:.0f}\nRX: {rx:.0f}\nLost: {lost:.0f}"
                        annot.set_text(text)
                        annot.set_visible(True)
                        self.enlarged_fig.canvas.draw_idle()
                        return
                annot.set_visible(False)
                self.enlarged_fig.canvas.draw_idle()
        
        if self.enlarged_fig is not None:
            self.enlarged_fig.canvas.mpl_connect("motion_notify_event", hover_bar)
    
    def plot_cdf_detailed(self, df, ax):
        """Detailed delay CDF"""
        ax.set_title('Cumulative Distribution Function (CDF) of Delay', fontsize=16, pad=20)
        ax.set_xlabel('Delay (ms)', fontsize=14, labelpad=10)
        ax.set_ylabel('CDF', fontsize=14, labelpad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        for service in ['eMBB', 'URLLC', 'mMTC']:
            service_data = df[df['service_type'] == service]
            if not service_data.empty:
                delays = service_data['avg_delay_ms'].dropna().values
                if len(delays) > 0:
                    sorted_delays = np.sort(delays)
                    cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
                    ax.plot(sorted_delays, cdf, label=service,
                           color=COLORS[service], linewidth=3, alpha=0.8)
        
        ax.legend(loc='lower right', fontsize=13, framealpha=0.9)
        ax.set_xlim(left=0)
        ax.set_ylim([0, 1])
        ax.tick_params(labelsize=12)
    
    def plot_jitter_detailed(self, df, ax):
        """Detailed jitter histogram"""
        ax.set_title('Jitter Distribution', fontsize=16, pad=20)
        ax.set_xlabel('Jitter (ms)', fontsize=14, labelpad=10)
        ax.set_ylabel('Frequency', fontsize=14, labelpad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        jitter_data = df['jitter_ms'].dropna().values
        if len(jitter_data) > 0:
            ax.hist(jitter_data, bins=40, color='#9b59b6', alpha=0.7, 
                   edgecolor='black', linewidth=1.2)
            
            # Add mean line
            mean_jitter = np.mean(jitter_data)
            ax.axvline(x=mean_jitter, color='#c0392b', linestyle='--', linewidth=3,
                      label=f'Mean: {mean_jitter:.3f}ms')
            ax.legend(fontsize=13, framealpha=0.9)
        
        ax.tick_params(labelsize=12)
    
    def plot_flows_detailed(self, df, ax):
        """Detailed active flows plot with hover tooltips"""
        ax.set_title('Active Flows Over Time', fontsize=16, pad=20)
        ax.set_xlabel('Time (s)', fontsize=14, labelpad=10)
        ax.set_ylabel('Number of Flows', fontsize=14, labelpad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        flows_per_time = df.groupby('timestamp')['flow_id'].nunique().reset_index()
        flows_per_time.columns = ['timestamp', 'flow_count']
        
        line, = ax.plot(flows_per_time['timestamp'], flows_per_time['flow_count'],
                       marker='o', color='#2c3e50', linewidth=3, markersize=7, alpha=0.8)
        ax.fill_between(flows_per_time['timestamp'], flows_per_time['flow_count'],
                       alpha=0.3, color='#3498db')
        
        ax.tick_params(labelsize=12)
        
        # Add hover annotation
        annot = ax.annotate("", xy=(0,0), xytext=(20,20),
                           textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.8", fc="#ecf0f1", ec="#34495e", lw=2, alpha=0.95),
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=2),
                           fontsize=11, weight='bold', color='#2c3e50')
        annot.set_visible(False)
        
        def hover_line(event):
            if event.inaxes == ax:
                cont, ind = line.contains(event)
                if cont:
                    idx = ind["ind"][0]
                    x = flows_per_time['timestamp'].iloc[idx]
                    y = flows_per_time['flow_count'].iloc[idx]
                    annot.xy = (x, y)
                    text = f"Time: {x:.2f}s\nActive Flows: {y}"
                    annot.set_text(text)
                    annot.set_visible(True)
                    self.enlarged_fig.canvas.draw_idle()
                    return
            annot.set_visible(False)
            self.enlarged_fig.canvas.draw_idle()
        
        if self.enlarged_fig is not None:
            self.enlarged_fig.canvas.mpl_connect("motion_notify_event", hover_line)

    def plot_cwnd_detailed(self, df, ax):
        """Detailed CWND plot for TCP flows with hover tooltips"""
        ax.set_title('TCP Congestion Window Evolution', fontsize=16, pad=20)
        ax.set_xlabel('Time (s)', fontsize=14, labelpad=10)
        ax.set_ylabel('CWND (segments)', fontsize=14, labelpad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        # Filter TCP flows (non-zero CWND)
        tcp_flows = df[df['cwnd'] > 0]
        
        if tcp_flows.empty:
            ax.text(0.5, 0.5, 'No TCP CWND data available\n(Only TCP flows show CWND)', 
                   ha='center', va='center', fontsize=14, color='#7f8c8d',
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Plot each TCP flow
        lines = []
        for flow_id in sorted(tcp_flows['flow_id'].unique()):
            flow_data = tcp_flows[tcp_flows['flow_id'] == flow_id]
            # Aggregate by timestamp (take max CWND per timestamp)
            agg_data = flow_data.groupby('timestamp')['cwnd'].max().reset_index()
            
            service = flow_data['service_type'].iloc[0]
            color = COLORS.get(service, COLORS['Unknown'])
            
            line, = ax.plot(agg_data['timestamp'], agg_data['cwnd'], 
                   label=f'Flow {flow_id} ({service})', 
                   color=color, linewidth=3, marker='o', markersize=5, alpha=0.8)
            lines.append((line, agg_data, flow_id, service))
        
        ax.legend(loc='upper left', fontsize=13, framealpha=0.9)
        ax.tick_params(labelsize=12)
        ax.set_ylim(bottom=0)
        
        # Add hover annotation
        annot = ax.annotate("", xy=(0,0), xytext=(20,20),
                           textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.8", fc="#ecf0f1", ec="#34495e", lw=2, alpha=0.95),
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=2),
                           fontsize=11, weight='bold', color='#2c3e50')
        annot.set_visible(False)
        
        def hover_line(event):
            if event.inaxes == ax:
                for line, data, flow_id, service in lines:
                    cont, ind = line.contains(event)
                    if cont:
                        idx = ind["ind"][0]
                        x = data['timestamp'].iloc[idx]
                        y = data['cwnd'].iloc[idx]
                        annot.xy = (x, y)
                        text = f"Flow {flow_id} ({service})\nTime: {x:.2f}s\nCWND: {y:.0f} segments"
                        annot.set_text(text)
                        annot.set_visible(True)
                        self.enlarged_fig.canvas.draw_idle()
                        return
                annot.set_visible(False)
                self.enlarged_fig.canvas.draw_idle()
        
        if self.enlarged_fig is not None:
            self.enlarged_fig.canvas.mpl_connect("motion_notify_event", hover_line)

    def load_csv_data(self):
        """Load and parse CSV metrics file"""
        try:
            if not os.path.exists(CSV_FILE):
                return None
            
            # Check if file has been updated
            current_size = os.path.getsize(CSV_FILE)
            if current_size == self.last_size and self.mode == 'realtime':
                return None
            self.last_size = current_size
            
            df = pd.read_csv(CSV_FILE)
            
            # Data validation
            if df.empty:
                return None
                
            # Convert data types
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['throughput_mbps'] = pd.to_numeric(df['throughput_mbps'], errors='coerce')
            df['avg_delay_ms'] = pd.to_numeric(df['avg_delay_ms'], errors='coerce')
            df['loss_rate'] = pd.to_numeric(df['loss_rate'], errors='coerce')
            df['jitter_ms'] = pd.to_numeric(df['jitter_ms'], errors='coerce')
            df['cwnd'] = pd.to_numeric(df['cwnd'], errors='coerce')  # NEW LINE

            return df
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None

    def safe_int(self, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def safe_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    
    def load_xml_data(self):
        """Parse FlowMonitor XML for detailed statistics"""
        try:
            if not os.path.exists(XML_FILE):
                return None
                
            tree = ET.parse(XML_FILE)
            root = tree.getroot()
            
            flows = []
            for flow in root.findall('.//Flow'):
                flow_data = {
                    'flowId': self.safe_int(flow.get('flowId')),
                    'txPackets': self.safe_int(flow.get('txPackets')),
                    'rxPackets': self.safe_int(flow.get('rxPackets')),
                    'lostPackets': self.safe_int(flow.get('lostPackets', 0)),
                    'txBytes': self.safe_int(flow.get('txBytes')),
                    'rxBytes': self.safe_int(flow.get('rxBytes')),
                    'delayMean': self.safe_float(flow.get('delayMean', '0').rstrip('ns')) / 1e6,
                    'jitterMean': self.safe_float(flow.get('jitterMean', '0').rstrip('ns')) / 1e6,
                }
                
                # Calculate metrics
                if flow_data['txPackets'] > 0:
                    flow_data['lossRate'] = (flow_data['lostPackets'] / flow_data['txPackets']) * 100
                else:
                    flow_data['lossRate'] = 0.0
                    
                flows.append(flow_data)
            
            return pd.DataFrame(flows)
            
        except Exception as e:
            print(f"Error loading XML: {e}")
            return None
    
    def update_plots(self, frame=None):
        """Update all plots with latest data"""
        df = self.load_csv_data()
        
        if df is None or df.empty:
            return
        
        # Update enlarged plot if active
        if self.enlarged_fig is not None and self.enlarged_ax is not None and self.enlarged_plot_type is not None:
            try:
                self.current_enlarged_data = df  # Update stored data
                self.update_enlarged_plot(df, self.enlarged_ax, self.enlarged_plot_type)
                self.enlarged_fig.canvas.draw_idle()
                self.enlarged_fig.canvas.flush_events()
            except Exception as e:
                print(f"Warning: Could not update enlarged plot: {e}")
        
        # Update main dashboard only if it's visible
        if self.main_fig is not None:
            try:
                # Clear all axes
                for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6, self.ax7]:
                    ax.clear()
            except Exception as e:
                print(f"Warning: Could not clear axes: {e}")
                return
            
            # Get latest timestamp for title
            latest_time = df['timestamp'].max()
            
            # ===== PLOT 1: Throughput Timeline =====
            self.ax1.set_title('Throughput Over Time', fontweight='bold', fontsize=11, pad=10)
            self.ax1.set_xlabel('Time (s)', fontsize=10)
            self.ax1.set_ylabel('Throughput (Mbps)', fontsize=10)
            self.ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            for service in ['eMBB', 'URLLC', 'mMTC', 'Background']:
                service_data = df[df['service_type'] == service]
                if not service_data.empty:
                    agg_data = service_data.groupby('timestamp')['throughput_mbps'].sum().reset_index()
                    self.ax1.plot(agg_data['timestamp'], agg_data['throughput_mbps'], 
                                 marker='o', label=service, color=COLORS[service],
                                 linewidth=2, markersize=4, alpha=0.7)
            
            self.ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
            
            # ===== PLOT 2: Service Distribution =====
            self.ax2.set_title('Traffic Distribution', fontweight='bold', fontsize=11, pad=10)

            # ✅ CORRECTED: Aggregate throughput calculation
            timestamps = df['timestamp'].unique()
            observation_window = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
            if observation_window < 0.1:
                observation_window = 1.0

            service_throughput = {}
            for service in ['eMBB', 'URLLC', 'mMTC', 'Background']:
                service_data = df[df['service_type'] == service]
                if not service_data.empty:
                    latest_time = service_data['timestamp'].max()
                    latest_data = service_data[service_data['timestamp'] == latest_time]
                    total_rx_packets = latest_data['rx_packets'].sum()
                    total_rx_bytes = total_rx_packets * 1400  # Approximate packet size
                    throughput_mbps = (total_rx_bytes * 8.0) / (observation_window * 1000000.0)
                    service_throughput[service] = throughput_mbps

            service_bytes = pd.Series(service_throughput)
            service_bytes = service_bytes[service_bytes > 0]

            if not service_bytes.empty:
                colors = [COLORS.get(s, COLORS['Unknown']) for s in service_bytes.index]
                explode = [0.05 if s == 'eMBB' else 0 for s in service_bytes.index]
                
                wedges, texts, autotexts = self.ax2.pie(
                    service_bytes.values, 
                    labels=None,
                    autopct='%1.1f%%',
                    colors=colors, 
                    explode=explode, 
                    startangle=90,
                    pctdistance=0.75,
                    textprops={'fontsize': 8}
                )
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_weight('bold')
                
                self.ax2.legend(
                    wedges, 
                    [f'{label}\n{value:.1f} Mbps' for label, value in zip(service_bytes.index, service_bytes.values)],
                    title="Services",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1),
                    fontsize=7,
                    title_fontsize=8
                )
            
            # ===== PLOT 3: Delay Timeline =====
            self.ax3.set_title('Average Delay Over Time', fontweight='bold', fontsize=11, pad=10)
            self.ax3.set_xlabel('Time (s)', fontsize=10)
            self.ax3.set_ylabel('Delay (ms)', fontsize=10)
            self.ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # URLLC threshold
            self.ax3.axhline(y=70, color='#c0392b', linestyle='--', linewidth=2, 
                            label='URLLC Target (<70ms)', alpha=0.7)
            
            for service in ['eMBB', 'URLLC', 'mMTC']:
                service_data = df[df['service_type'] == service]
                if not service_data.empty:
                    agg_delay = service_data.groupby('timestamp')['avg_delay_ms'].mean().reset_index()
                    self.ax3.plot(agg_delay['timestamp'], agg_delay['avg_delay_ms'],
                                marker='o', label=service, color=COLORS[service],
                                linewidth=2, markersize=4, alpha=0.7)
            
            self.ax3.legend(loc='upper left', fontsize=8, framealpha=0.9)
            
            # ===== PLOT 4: Loss Rate =====
            self.ax4.set_title('Packet Loss Rate', fontweight='bold', fontsize=11, pad=10)
            self.ax4.set_ylabel('Loss Rate (%)', fontsize=10)
            
            loss_by_service = df.groupby('service_type')['loss_rate'].mean()
            if not loss_by_service.empty:
                colors_loss = [COLORS.get(s, COLORS['Unknown']) for s in loss_by_service.index]
                bars = self.ax4.bar(range(len(loss_by_service)), loss_by_service.values,
                                   color=colors_loss, alpha=0.7, edgecolor='black', linewidth=1)
                
                self.ax4.set_xticks(range(len(loss_by_service)))
                self.ax4.set_xticklabels(loss_by_service.index, rotation=45, fontsize=8)
                self.ax4.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    self.ax4.text(bar.get_x() + bar.get_width()/2., height,
                                 f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # ===== PLOT 5: Delay CDF =====
            self.ax5.set_title('Delay CDF', fontweight='bold', fontsize=11, pad=10)
            self.ax5.set_xlabel('Delay (ms)', fontsize=10)
            self.ax5.set_ylabel('CDF', fontsize=10)
            self.ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            for service in ['eMBB', 'URLLC', 'mMTC']:
                service_data = df[df['service_type'] == service]
                if not service_data.empty:
                    delays = service_data['avg_delay_ms'].dropna().values
                    if len(delays) > 0:
                        sorted_delays = np.sort(delays)
                        cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
                        self.ax5.plot(sorted_delays, cdf, label=service,
                                     color=COLORS[service], linewidth=2, alpha=0.7)
            
            self.ax5.legend(loc='lower right', fontsize=8, framealpha=0.9)
            self.ax5.set_xlim(left=0)
            self.ax5.set_ylim([0, 1])
            
            # ===== PLOT 6: Jitter Histogram =====
            self.ax6.set_title('Jitter Distribution', fontweight='bold', fontsize=11, pad=10)
            self.ax6.set_xlabel('Jitter (ms)', fontsize=10)
            self.ax6.set_ylabel('Frequency', fontsize=10)
            self.ax6.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            jitter_data = df['jitter_ms'].dropna().values
            if len(jitter_data) > 0:
                self.ax6.hist(jitter_data, bins=25, color='#9b59b6', alpha=0.7, 
                             edgecolor='black', linewidth=0.8)
                
                mean_jitter = np.mean(jitter_data)
                self.ax6.axvline(x=mean_jitter, color='#c0392b', linestyle='--', linewidth=2,
                                label=f'Mean: {mean_jitter:.2f}ms')
                self.ax6.legend(fontsize=8, framealpha=0.9)
            
            # ===== PLOT 7: TCP CWND Evolution =====
            self.ax7.set_title('TCP CWND Evolution', fontweight='bold', fontsize=11, pad=10)
            self.ax7.set_xlabel('Time (s)', fontsize=10)
            self.ax7.set_ylabel('CWND (segments)', fontsize=10)
            self.ax7.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Filter TCP flows (non-zero CWND)
            tcp_flows = df[df['cwnd'] > 0]
            
            if not tcp_flows.empty:
                # Plot each TCP flow
                for flow_id in sorted(tcp_flows['flow_id'].unique()):
                    flow_data = tcp_flows[tcp_flows['flow_id'] == flow_id]
                    # Aggregate by timestamp (take max CWND per timestamp)
                    agg_data = flow_data.groupby('timestamp')['cwnd'].max().reset_index()
                    
                    service = flow_data['service_type'].iloc[0]
                    color = COLORS.get(service, COLORS['Unknown'])
                    
                    self.ax7.plot(agg_data['timestamp'], agg_data['cwnd'], 
                           label=f'Flow {flow_id}', 
                           color=color, linewidth=2, marker='o', markersize=3, alpha=0.7)
                
                self.ax7.legend(loc='upper left', fontsize=7, framealpha=0.9)
                self.ax7.set_ylim(bottom=0)
            else:
                self.ax7.text(0.5, 0.5, 'No TCP flows\n(CWND not available)', 
                       ha='center', va='center', transform=self.ax7.transAxes,
                       fontsize=9, color='#7f8c8d')
            
            # Add timestamp for real-time mode
            if self.mode == 'realtime':
                self.main_fig.text(0.99, 0.005, 
                                  f'Last Update: {datetime.now().strftime("%H : %M : %S")}',
                                  ha='right', va='bottom', fontsize=9, 
                                  style='italic', color='#7f8c8d')
            
            try:
                self.main_fig.canvas.draw_idle()
                self.main_fig.canvas.flush_events()
            except Exception as e:
                print(f"Warning: Could not redraw main figure: {e}")
    
    def run_realtime(self):
        """Run real-time monitoring"""
        print("\n" + "="*70)
        print("    REAL-TIME NETWORK METRICS MONITOR")
        print("="*70)
        print("\n  📊 Monitoring: mec_metrics.csv")
        print("  🔄 Update Interval: 2 seconds")
        print("  🖱️  Click any subplot to enlarge")
        print("  ⌨️  Press ESC in enlarged view to return")
        print("  ⏹️  Press Ctrl+C to stop\n")
        print("="*70 + "\n")
        
        # Check if CSV exists
        if not os.path.exists(CSV_FILE):
            print(f"⚠️  Waiting for {CSV_FILE} to be created...")
            print("   (Run NS-3 simulation first)\n")
            
            # Wait for file to appear
            while not os.path.exists(CSV_FILE):
                time.sleep(1)
            
            print(f"✅ {CSV_FILE} detected! Starting visualization...\n")
            time.sleep(2)
        
        # Setup animation
        ani = animation.FuncAnimation(self.main_fig, self.update_plots,
                                     interval=UPDATE_INTERVAL, blit=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n\n✅ Real-time monitoring stopped.")
    
    def run_postsim(self):
        """Run post-simulation analysis"""
        print("\n" + "="*70)
        print("    POST-SIMULATION NETWORK METRICS ANALYSIS")
        print("="*70 + "\n")
        
        # Check files exist
        if not os.path.exists(CSV_FILE):
            print(f"❌ Error: {CSV_FILE} not found!")
            print("   Run NS-3 simulation first to generate metrics.\n")
            return
        
        print(f"📊 Loading data from {CSV_FILE}...")
        df = self.load_csv_data()
        
        if df is None or df.empty:
            print("❌ No data found in CSV file!")
            return
        
        print(f"✅ Loaded {len(df)} data points")
        print(f"   Flows: {df['flow_id'].nunique()}")
        print(f"   Time range: {df['timestamp'].min():.1f}s - {df['timestamp'].max():.1f}s\n")
        
        # Generate comprehensive statistics
        self.print_statistics(df)
        
        # Update plots once with all data
        print("🎨 Generating visualizations...")
        print("   💡 Click any subplot to view enlarged")
        print("   💡 Press ESC in enlarged view to return\n")
        self.update_plots()
        
        # Load XML data if available
        if os.path.exists(XML_FILE):
            print(f"📄 Loading FlowMonitor data from {XML_FILE}...")
            xml_df = self.load_xml_data()
            if xml_df is not None:
                self.print_xml_statistics(xml_df)
        
        print("\n" + "="*70)
        print("✅ Analysis complete! Dashboard ready.")
        print("="*70 + "\n")
        plt.show()
    
    def print_statistics(self, df):
        """Print detailed statistics from CSV data"""
        print("\n" + "━"*70)
        print("                    OVERALL STATISTICS")
        print("━"*70 + "\n")
        
        print(f"📈 Total Flows Analyzed: {df['flow_id'].nunique()}")
        print(f"⏱️  Simulation Duration: {df['timestamp'].min():.1f}s - {df['timestamp'].max():.1f}s")
        print(f"📦 Total Packets TX: {df['tx_packets'].sum():.0f}")
        print(f"📥 Total Packets RX: {df['rx_packets'].sum():.0f}")
        print(f"❌ Total Packets Lost: {df['lost_packets'].sum():.0f}")
        
        overall_loss = (df['lost_packets'].sum() / df['tx_packets'].sum() * 100) if df['tx_packets'].sum() > 0 else 0
        print(f"📉 Overall Loss Rate: {overall_loss:.2f}%")
        
        print("\n" + "─"*70)
        print("               PER-SERVICE TYPE BREAKDOWN")
        print("─"*70)
        
        for service in ['eMBB', 'URLLC', 'mMTC', 'Background']:
            service_data = df[df['service_type'] == service]
            if not service_data.empty:
                print(f"\n🔹 {service}:")
                print(f"   Flows: {service_data['flow_id'].nunique()}")
                print(f"   Avg Throughput: {service_data['throughput_mbps'].mean():.2f} Mbps")
                print(f"   Avg Delay: {service_data['avg_delay_ms'].mean():.2f} ms")
                print(f"   Avg Jitter: {service_data['jitter_ms'].mean():.2f} ms")
                print(f"   Loss Rate: {service_data['loss_rate'].mean():.2f}%")
                
                # URLLC SLA check
                if service == 'URLLC':
                    sla_violations = (service_data['avg_delay_ms'] > 70).sum()
                    total_samples = len(service_data)
                    sla_compliance = ((total_samples - sla_violations) / total_samples * 100) if total_samples > 0 else 0
                    print(f"   ⚡ SLA Compliance (<70ms): {sla_compliance:.1f}%")
                    if sla_compliance < 90:
                        print(f"      ⚠️  WARNING: SLA violations detected!")
        
        print("\n" + "━"*70 + "\n")
    
    def print_xml_statistics(self, xml_df):
        """Print statistics from FlowMonitor XML"""
        print("\n" + "━"*70)
        print("              FLOWMONITOR DETAILED ANALYSIS")
        print("━"*70 + "\n")
        
        print(f"📊 Total Flows: {len(xml_df)}")
        print(f"📦 Total TX Packets: {xml_df['txPackets'].sum()}")
        print(f"📥 Total RX Packets: {xml_df['rxPackets'].sum()}")
        print(f"❌ Total Lost Packets: {xml_df['lostPackets'].sum()}")
        print(f"📉 Overall Loss Rate: {xml_df['lossRate'].mean():.2f}%")
        
        print(f"\n⏱️  Delay Statistics:")
        print(f"   Mean: {xml_df['delayMean'].mean():.2f} ms")
        print(f"   Min: {xml_df['delayMean'].min():.2f} ms")
        print(f"   Max: {xml_df['delayMean'].max():.2f} ms")
        print(f"   Std Dev: {xml_df['delayMean'].std():.2f} ms")
        
        print(f"\n📶 Jitter Statistics:")
        print(f"   Mean: {xml_df['jitterMean'].mean():.2f} ms")
        print(f"   Min: {xml_df['jitterMean'].min():.2f} ms")
        print(f"   Max: {xml_df['jitterMean'].max():.2f} ms")
        
        # Top 5 flows by throughput
        print(f"\n🏆 Top 5 Flows by Data Volume:")
        top_flows = xml_df.nlargest(5, 'rxBytes')[['flowId', 'rxBytes', 'lossRate', 'delayMean']]
        for idx, row in top_flows.iterrows():
            print(f"   Flow {row['flowId']}: {row['rxBytes']/1e6:.2f} MB, "
                  f"Loss: {row['lossRate']:.2f}%, Delay: {row['delayMean']:.2f} ms")
        
        print("\n" + "━"*70 + "\n")


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("    5G MEC NETWORK METRICS VISUALIZATION TOOL")
    print("    Enhanced Interactive Dashboard")
    print("="*70 + "\n")
    
    # ✅ NEW: Add option for metrics report
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'report':
            # Generate network metrics report only
            analyzer = NetworkMetricsAnalyzer(CSV_FILE)
            metrics = analyzer.print_report()
            analyzer.save_summary()
            return
        
        mode = sys.argv[1].lower()
    else:
        print("Select mode:")
        print("  1. Real-time monitoring (during simulation)")
        print("  2. Post-simulation analysis (after simulation)")
        print("  3. Network metrics report (text summary)")  # ✅ NEW
        print("\nChoice (1/2/3): ", end='')
        
        choice = input().strip()
        if choice == '3':
            analyzer = NetworkMetricsAnalyzer(CSV_FILE)
            metrics = analyzer.print_report()
            analyzer.save_summary()
            return
        
        mode = 'realtime' if choice == '1' else 'postsim'
    
    # ... rest of existing code ...
    
    visualizer = MetricsVisualizer(mode=mode)
    
    if mode == 'realtime':
        visualizer.run_realtime()
    else:
        visualizer.run_postsim()

# ============================================================================
# NETWORK METRICS ANALYZER (FOR RESULT COMPARISON)
# ============================================================================
class NetworkMetricsAnalyzer:
    """
    Extract and display network-only metrics for TCP comparison
    (No RL/reward metrics - pure network performance)
    """
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.df = None
    
    def load_data(self):
        """Load CSV data"""
        try:
            self.df = pd.read_csv(self.csv_file)
            self.df['timestamp'] = pd.to_numeric(self.df['timestamp'], errors='coerce')
            self.df['throughput_mbps'] = pd.to_numeric(self.df['throughput_mbps'], errors='coerce')
            self.df['avg_delay_ms'] = pd.to_numeric(self.df['avg_delay_ms'], errors='coerce')
            self.df['loss_rate'] = pd.to_numeric(self.df['loss_rate'], errors='coerce')
            self.df['jitter_ms'] = pd.to_numeric(self.df['jitter_ms'], errors='coerce')
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def compute_metrics(self) -> Dict:
        """Compute all network metrics"""
        if self.df is None or self.df.empty:
            return {}
        
        metrics = {}
        
        # ===== 1. THROUGHPUT METRICS =====
        # Aggregate throughput (sum across all flows at each timestamp)
        agg_throughput = self.df.groupby('timestamp')['throughput_mbps'].sum()
        metrics['aggregate_throughput_mean'] = agg_throughput.mean()
        metrics['aggregate_throughput_std'] = agg_throughput.std()
        metrics['aggregate_throughput_max'] = agg_throughput.max()
        
        # Per-flow average throughput
        per_flow_avg = self.df.groupby('flow_id')['throughput_mbps'].mean()
        metrics['per_flow_throughput_mean'] = per_flow_avg.mean()
        metrics['per_flow_throughput_std'] = per_flow_avg.std()
        
        # ===== 2. LATENCY METRICS =====
        delays = self.df['avg_delay_ms'].dropna()
        metrics['latency_mean'] = delays.mean()
        metrics['latency_std'] = delays.std()
        metrics['latency_min'] = delays.min()
        metrics['latency_max'] = delays.max()
        metrics['latency_p50'] = delays.quantile(0.50)
        metrics['latency_p95'] = delays.quantile(0.95)
        metrics['latency_p99'] = delays.quantile(0.99)
        
        # ===== 3. PACKET LOSS RATE =====
        metrics['loss_rate_mean'] = self.df['loss_rate'].mean()
        metrics['loss_rate_max'] = self.df['loss_rate'].max()
        
        total_tx = self.df['tx_packets'].sum()
        total_rx = self.df['rx_packets'].sum()
        total_lost = self.df['lost_packets'].sum()
        metrics['overall_loss_rate'] = (total_lost / total_tx * 100) if total_tx > 0 else 0
        
        # ===== 4. SLA VIOLATION RATE =====
        # Get latest timestamp data for each flow
        latest_time = self.df['timestamp'].max()
        latest_data = self.df[self.df['timestamp'] == latest_time]
        
        total_flows = latest_data['flow_id'].nunique()
        
        # URLLC violations (RTT > 70ms OR loss > 1%)
        urllc_flows = latest_data[latest_data['service_type'] == 'URLLC']
        urllc_violations = ((urllc_flows['avg_delay_ms'] > 70) | 
                           (urllc_flows['loss_rate'] > 1.0)).sum()
        
        # eMBB violations (throughput < 5 Mbps)
        embb_flows = latest_data[latest_data['service_type'] == 'eMBB']
        embb_violations = (embb_flows['throughput_mbps'] < 5.0).sum()
        
        # mMTC violations (loss > 5%)
        mmtc_flows = latest_data[latest_data['service_type'] == 'mMTC']
        mmtc_violations = (mmtc_flows['loss_rate'] > 5.0).sum()
        
        total_violations = urllc_violations + embb_violations + mmtc_violations
        
        metrics['sla_violation_rate'] = (total_violations / total_flows * 100) if total_flows > 0 else 0
        metrics['urllc_violation_count'] = urllc_violations
        metrics['embb_violation_count'] = embb_violations
        metrics['mmtc_violation_count'] = mmtc_violations
        
        # ===== 5. FAIRNESS INDEX (JAIN'S) - PER SERVICE TYPE =====
        latest_time = self.df['timestamp'].max()
        latest_data = self.df[self.df['timestamp'] == latest_time]

        # Calculate fairness per service type
        for service in ['eMBB', 'URLLC', 'mMTC', 'Background']:
            service_flows = latest_data[latest_data['service_type'] == service]
            throughputs = service_flows['throughput_mbps'].values
            
            if len(throughputs) > 1:
                sum_x = np.sum(throughputs)
                sum_x2 = np.sum(throughputs ** 2)
                fairness = (sum_x ** 2) / (len(throughputs) * sum_x2) if sum_x2 > 0 else 0
                metrics[f'fairness_index_{service.lower()}'] = fairness
            else:
                metrics[f'fairness_index_{service.lower()}'] = 1.0

        # Overall fairness (weighted average across service types)
        fairness_values = [metrics.get(f'fairness_index_{s.lower()}', 1.0) 
                        for s in ['eMBB', 'URLLC', 'mMTC', 'Background']]
        metrics['fairness_index_overall'] = np.mean(fairness_values)
        
        return metrics
    
    def print_report(self):
        """Print comprehensive network metrics report"""
        if not self.load_data():
            print("Failed to load data")
            return
        
        metrics = self.compute_metrics()
        
        print("\n" + "="*70)
        print("           NETWORK PERFORMANCE METRICS REPORT")
        print("="*70)
        
        print("\n📊 1. THROUGHPUT METRICS")
        print("-" * 70)
        print(f"  Aggregate Throughput (Mean):    {metrics['aggregate_throughput_mean']:8.2f} Mbps")
        print(f"  Aggregate Throughput (Std Dev): {metrics['aggregate_throughput_std']:8.2f} Mbps")
        print(f"  Aggregate Throughput (Peak):    {metrics['aggregate_throughput_max']:8.2f} Mbps")
        print(f"  Per-Flow Throughput (Mean):     {metrics['per_flow_throughput_mean']:8.2f} Mbps")
        print(f"  Per-Flow Throughput (Std Dev):  {metrics['per_flow_throughput_std']:8.2f} Mbps")
        
        print("\n⏱️  2. LATENCY METRICS")
        print("-" * 70)
        print(f"  Mean Latency:                   {metrics['latency_mean']:8.2f} ms")
        print(f"  Std Dev:                        {metrics['latency_std']:8.2f} ms")
        print(f"  Min Latency:                    {metrics['latency_min']:8.2f} ms")
        print(f"  Max Latency:                    {metrics['latency_max']:8.2f} ms")
        print(f"  50th Percentile (Median):       {metrics['latency_p50']:8.2f} ms")
        print(f"  95th Percentile:                {metrics['latency_p95']:8.2f} ms")
        print(f"  99th Percentile:                {metrics['latency_p99']:8.2f} ms")
        
        print("\n📉 3. PACKET LOSS METRICS")
        print("-" * 70)
        print(f"  Overall Loss Rate:              {metrics['overall_loss_rate']:8.2f} %")
        print(f"  Mean Loss Rate (per flow):      {metrics['loss_rate_mean']:8.2f} %")
        print(f"  Max Loss Rate:                  {metrics['loss_rate_max']:8.2f} %")
        
        print("\n🎯 4. SLA COMPLIANCE")
        print("-" * 70)
        print(f"  Overall SLA Violation Rate:     {metrics['sla_violation_rate']:8.2f} %")
        print(f"  URLLC Violations:               {metrics['urllc_violation_count']:8.0f}")
        print(f"  eMBB Violations:                {metrics['embb_violation_count']:8.0f}")
        print(f"  mMTC Violations:                {metrics['mmtc_violation_count']:8.0f}")
        
        print("\n⚖️  5. FAIRNESS (PER SERVICE TYPE)")
        print("-" * 70)
        print(f"  eMBB Fairness Index:            {metrics.get('fairness_index_embb', 1.0):8.4f}")
        print(f"  URLLC Fairness Index:           {metrics.get('fairness_index_urllc', 1.0):8.4f}")
        print(f"  mMTC Fairness Index:            {metrics.get('fairness_index_mmtc', 1.0):8.4f}")
        print(f"  Background Fairness Index:      {metrics.get('fairness_index_background', 1.0):8.4f}")
        print(f"  Overall (Weighted Avg):         {metrics['fairness_index_overall']:8.4f}")
        print(f"                                  (1.0 = perfect fairness within type)")
        
        print("\n" + "="*70 + "\n")
        
        return metrics
    
    def save_summary(self, output_file: str = "network_metrics_summary.txt"):
        """Save metrics to text file"""
        metrics = self.compute_metrics()
        
        with open(output_file, 'w') as f:
            f.write("NETWORK PERFORMANCE METRICS\n")
            f.write("="*70 + "\n\n")
            
            f.write("THROUGHPUT:\n")
            f.write(f"  Aggregate Mean: {metrics['aggregate_throughput_mean']:.2f} Mbps\n")
            f.write(f"  Aggregate Peak: {metrics['aggregate_throughput_max']:.2f} Mbps\n")
            f.write(f"  Per-Flow Mean:  {metrics['per_flow_throughput_mean']:.2f} Mbps\n\n")
            
            f.write("LATENCY:\n")
            f.write(f"  Mean: {metrics['latency_mean']:.2f} ms\n")
            f.write(f"  P50:  {metrics['latency_p50']:.2f} ms\n")
            f.write(f"  P95:  {metrics['latency_p95']:.2f} ms\n")
            f.write(f"  P99:  {metrics['latency_p99']:.2f} ms\n\n")
            
            f.write("PACKET LOSS:\n")
            f.write(f"  Overall: {metrics['overall_loss_rate']:.2f} %\n\n")
            
            f.write("SLA COMPLIANCE:\n")
            f.write(f"  Violation Rate: {metrics['sla_violation_rate']:.2f} %\n")
            f.write(f"  URLLC Violations: {metrics['urllc_violation_count']:.0f}\n")
            f.write(f"  eMBB Violations:  {metrics['embb_violation_count']:.0f}\n\n")
            
            f.write("FAIRNESS (PER SERVICE TYPE):\n")
            f.write(f"  eMBB:       {metrics.get('fairness_index_embb', 1.0):.4f}\n")
            f.write(f"  URLLC:      {metrics.get('fairness_index_urllc', 1.0):.4f}\n")
            f.write(f"  mMTC:       {metrics.get('fairness_index_mmtc', 1.0):.4f}\n")
            f.write(f"  Background: {metrics.get('fairness_index_background', 1.0):.4f}\n")
            f.write(f"  Overall:    {metrics['fairness_index_overall']:.4f}\n")
        
        print(f"✅ Summary saved to {output_file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ Visualization stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
