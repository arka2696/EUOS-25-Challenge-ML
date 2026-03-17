"""Exploratory data analysis (EDA) subpackage for EUOS25.

This subpackage groups together plotting utilities for understanding the
distribution and relationships of molecular features and labels in the
EUOS25 dataset.
"""

from .plots import (
    plot_class_distribution,
    plot_descriptor_histograms,
    plot_correlation_heatmap,
    tsne_projection,
    functional_group_stats,
)

__all__ = [
    "plots",
    "plot_class_distribution",
    "plot_descriptor_histograms",
    "plot_correlation_heatmap",
    "tsne_projection",
    "functional_group_stats",
]
