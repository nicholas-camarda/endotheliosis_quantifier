# Spec Summary (Lite)

Implement dual-environment architecture for endotheliosis quantifier package with explicit mode selection between production (RTX 3080/CUDA) and development (M1 Mac/Metal) workflows. Design hardware capability detection, automatic suggestions, and explicit user control over mode selection while maintaining unified CLI interface for seamless environment transitions.

**Additionally, implement comprehensive 3-stage pipeline organization and enhanced visualization capabilities** including organized output directories with data-driven naming, multi-level visualization pipeline showing progression through segmentation training, quantification training, and production inference, comprehensive reporting with interactive visualizations, and ROI extraction and feature computation visualization.

**Status**: Pipeline organization and CLI consolidation completed. Next: Implement 3-stage pipeline architecture (seg, quant-endo, production).

**Current CLI**: `python -m eq orchestrator` (interactive menu) or `python -m eq pipeline` (full pipeline)

**Planned CLI**: 
- `python -m eq seg` (segmentation training)
- `python -m eq quant-endo` (quantification training) 
- `python -m eq production` (end-to-end inference)
- All commands support `QUICK_TEST=true` for fast validation mode
