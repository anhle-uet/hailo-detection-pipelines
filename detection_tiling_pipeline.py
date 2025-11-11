#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tiling with Resolution-Preserving Hailo Detection Pipeline
----------------------------------------------------------
Uses hailotilecropper + hailotileaggregator to split frame into tiles,
process each tile through the neural network, and stitch results back
while preserving original resolution.
"""

import os
import sys
import argparse
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

# ============================================================================
# CONFIGURATION - Edit these paths
# ============================================================================

DEFAULT_HEF_PATH = "/usr/local/hailo/resources/models/hailo8/yolov11m.hef"
DEFAULT_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"

# Tiling configuration
TILES_X = 2  # Number of tiles along x-axis (columns)
TILES_Y = 2  # Number of tiles along y-axis (rows)
OVERLAP_X = 0.2  # Overlap percentage between tiles along x-axis (0.0 - 1.0)
OVERLAP_Y = 0.2  # Overlap percentage between tiles along y-axis (0.0 - 1.0)
TILING_MODE = 0  # 0: single-scale, 1: multi-scale

# Tile processing resolution (model input size)
TILE_WIDTH = 640
TILE_HEIGHT = 640

# NMS parameters for aggregator
IOU_THRESHOLD = 0.3  # NMS IOU threshold for removing duplicate detections
BORDER_THRESHOLD = 0.1  # Remove detections near tile borders (0.0 - 1.0)
REMOVE_LARGE_LANDSCAPE = True  # Remove large landscape objects in multi-scale mode

# Encoding parameters
BITRATE = 4000  # kbps
ENCODING_PRESET = "ultrafast"

# ============================================================================

Gst.init(None)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tiling pipeline with resolution preservation for Hailo detection"
    )
    parser.add_argument("--input", required=True, help="Input MP4 file path")
    parser.add_argument("--output", required=True, help="Output MP4 file path")
    parser.add_argument("--hef", default=DEFAULT_HEF_PATH, help="HEF model file path")
    parser.add_argument("--postprocess-so", default=DEFAULT_POSTPROCESS_SO,
                       help="Postprocess .so library path")
    
    # Tiling parameters
    parser.add_argument("--tiles-x", type=int, default=TILES_X,
                       help="Number of tiles along x-axis (columns)")
    parser.add_argument("--tiles-y", type=int, default=TILES_Y,
                       help="Number of tiles along y-axis (rows)")
    parser.add_argument("--overlap-x", type=float, default=OVERLAP_X,
                       help="Overlap percentage between tiles along x-axis (0.0-1.0)")
    parser.add_argument("--overlap-y", type=float, default=OVERLAP_Y,
                       help="Overlap percentage between tiles along y-axis (0.0-1.0)")
    parser.add_argument("--tiling-mode", type=int, default=TILING_MODE,
                       help="Tiling mode (0: single-scale, 1: multi-scale)")
    
    # Tile processing resolution
    parser.add_argument("--tile-width", type=int, default=TILE_WIDTH,
                       help="Tile processing width (model input)")
    parser.add_argument("--tile-height", type=int, default=TILE_HEIGHT,
                       help="Tile processing height (model input)")
    
    # Aggregator parameters
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD,
                       help="NMS IOU threshold for aggregator (0.0-1.0)")
    parser.add_argument("--border-threshold", type=float, default=BORDER_THRESHOLD,
                       help="Border threshold for removing edge detections (0.0-1.0)")
    parser.add_argument("--remove-large-landscape", action="store_true",
                       default=REMOVE_LARGE_LANDSCAPE,
                       help="Remove large landscape objects in multi-scale mode")
    
    # Encoding parameters
    parser.add_argument("--bitrate", type=int, default=BITRATE,
                       help="Output video bitrate in kbps")
    
    parser.add_argument("--debug", action="store_true",
                       help="Print debug information")
    return parser.parse_args()


class TilingResolutionPreservingPipeline:
    def __init__(self, args):
        self.args = args
        self.pipeline = None
        self.loop = None
        self.frame_count = 0
        
        # Validate paths and parameters
        self._validate_paths()
        self._validate_parameters()
        
    def _validate_paths(self):
        """Validate that required files exist"""
        if not Path(self.args.input).exists():
            raise FileNotFoundError(f"Input file not found: {self.args.input}")
        
        if not Path(self.args.hef).exists():
            raise FileNotFoundError(f"HEF file not found: {self.args.hef}")
        
        if not Path(self.args.postprocess_so).exists():
            raise FileNotFoundError(f"Postprocess .so not found: {self.args.postprocess_so}")
        
        # Create output directory if needed
        output_dir = Path(self.args.output).parent
        if output_dir != Path('.'):
            output_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_parameters(self):
        """Validate tiling and aggregator parameters"""
        if not (1 <= self.args.tiles_x <= 20):
            raise ValueError(f"tiles-x must be between 1 and 20, got {self.args.tiles_x}")
        
        if not (1 <= self.args.tiles_y <= 20):
            raise ValueError(f"tiles-y must be between 1 and 20, got {self.args.tiles_y}")
        
        if not (0.0 <= self.args.overlap_x <= 1.0):
            raise ValueError(f"overlap-x must be between 0.0 and 1.0, got {self.args.overlap_x}")
        
        if not (0.0 <= self.args.overlap_y <= 1.0):
            raise ValueError(f"overlap-y must be between 0.0 and 1.0, got {self.args.overlap_y}")
        
        if self.args.tiling_mode not in [0, 1]:
            raise ValueError(f"tiling-mode must be 0 or 1, got {self.args.tiling_mode}")
        
        if not (0.0 <= self.args.iou_threshold <= 1.0):
            raise ValueError(f"iou-threshold must be between 0.0 and 1.0, got {self.args.iou_threshold}")
        
        if not (0.0 <= self.args.border_threshold <= 1.0):
            raise ValueError(f"border-threshold must be between 0.0 and 1.0, got {self.args.border_threshold}")
    
    def build_pipeline(self):
        """
        Build GStreamer tiling pipeline with resolution preservation.
        
        Pipeline structure:
        1. Source -> decode -> convert
        2. hailotilecropper splits frame into:
           a) Original resolution (src_0) -> aggregator
           b) Tiles (src_1) -> rescale -> hailonet -> hailofilter -> aggregator
        3. hailotileaggregator stitches results back at original resolution
        4. hailooverlay -> encode -> save
        """
        
        tiling_mode_str = "single-scale" if self.args.tiling_mode == 0 else "multi-scale"
        
        pipeline_str = f"""
            filesrc location="{self.args.input}" !
            qtdemux !
            h264parse !
            avdec_h264 !
            videoconvert !
            video/x-raw,format=RGB !
            hailotilecropper name=cropper
                tiles-along-x-axis={self.args.tiles_x}
                tiles-along-y-axis={self.args.tiles_y}
                overlap-x-axis={self.args.overlap_x}
                overlap-y-axis={self.args.overlap_y}
                tiling-mode={self.args.tiling_mode}
            
            hailotileaggregator name=aggregator
                flatten-detections=true
                iou-threshold={self.args.iou_threshold}
                border-threshold={self.args.border_threshold}
                remove-large-landscape={str(self.args.remove_large_landscape).lower()}
            
            cropper.src_0 !
            queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
            aggregator.sink_0
            
            cropper.src_1 !
            queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
            videoscale !
            video/x-raw,width={self.args.tile_width},height={self.args.tile_height} !
            queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
            hailonet hef-path={self.args.hef} is-active=true !
            queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
            hailofilter so-path={self.args.postprocess_so} qos=false !
            queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
            aggregator.sink_1
            
            aggregator.src !
            queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
            hailooverlay !
            queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
            videoconvert !
            x264enc bitrate={self.args.bitrate} speed-preset={ENCODING_PRESET} tune=zerolatency key-int-max=60 !
            h264parse !
            mp4mux !
            filesink location="{self.args.output}" sync=false
        """
        
        # Clean up the string
        pipeline_str = " ".join(pipeline_str.split())
        
        if self.args.debug:
            print("\n[DEBUG] GStreamer Tiling Pipeline:")
            print("-" * 80)
            formatted = pipeline_str.replace(" ! ", " !\n    ")
            print(formatted)
            print("-" * 80)
            print()
        
        return pipeline_str
    
    def on_message(self, bus, message):
        """Handle GStreamer bus messages"""
        mtype = message.type
        
        if mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\n[ERROR] {err}")
            if debug and self.args.debug:
                print(f"[DEBUG] {debug}")
            if self.loop:
                self.loop.quit()
        
        elif mtype == Gst.MessageType.EOS:
            print(f"\n[EOS] Processing complete. Processed {self.frame_count} frames.")
            if self.loop:
                self.loop.quit()
        
        elif mtype == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"[WARNING] {warn}")
            if debug and self.args.debug:
                print(f"[DEBUG] {debug}")
        
        elif mtype == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending = message.parse_state_changed()
                if self.args.debug:
                    print(f"[STATE] {old_state.value_nick} -> {new_state.value_nick}")
        
        elif mtype == Gst.MessageType.STREAM_STATUS:
            if self.args.debug:
                print(f"[STREAM_STATUS] {message.parse_stream_status()}")
        
        return True
    
    def on_buffer_probe(self, pad, info):
        """Monitor processing progress"""
        self.frame_count += 1
        
        if self.frame_count % 50 == 0:
            print(f"[PROGRESS] Processed {self.frame_count} frames...", end='\r')
        
        return Gst.PadProbeReturn.OK
    
    def run(self):
        """Build and run the tiling pipeline"""
        print("\n" + "=" * 80)
        print("Tiling Resolution-Preserving Hailo Detection Pipeline")
        print("=" * 80)
        print(f"Input:              {self.args.input}")
        print(f"Output:             {self.args.output}")
        print(f"HEF:                {self.args.hef}")
        print(f"Postprocess:        {self.args.postprocess_so}")
        print("-" * 80)
        print(f"Tiling grid:        {self.args.tiles_x} x {self.args.tiles_y}")
        print(f"Tile overlap:       X={self.args.overlap_x:.1%}, Y={self.args.overlap_y:.1%}")
        print(f"Tiling mode:        {'Single-scale' if self.args.tiling_mode == 0 else 'Multi-scale'}")
        print(f"Tile process size:  {self.args.tile_width}x{self.args.tile_height}")
        print("-" * 80)
        print(f"IOU threshold:      {self.args.iou_threshold}")
        print(f"Border threshold:   {self.args.border_threshold}")
        print(f"Remove large:       {self.args.remove_large_landscape}")
        print(f"Output bitrate:     {self.args.bitrate} kbps")
        print("=" * 80)
        print()
        
        # Build pipeline
        try:
            pipeline_str = self.build_pipeline()
            self.pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            print(f"[ERROR] Failed to create pipeline: {e}")
            return 1
        
        # Add buffer probe to monitor progress
        aggregator = self.pipeline.get_by_name("aggregator")
        if aggregator:
            src_pad = aggregator.get_static_pad("src")
            if src_pad:
                src_pad.add_probe(Gst.PadProbeType.BUFFER, self.on_buffer_probe)
        
        # Setup message bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)
        
        # Create main loop
        self.loop = GLib.MainLoop()
        
        # Start pipeline
        print("[START] Starting tiling pipeline...\n")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] Unable to set pipeline to PLAYING state")
            return 1
        
        # Run
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Stopping pipeline...")
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            print(f"\n[DONE] Output saved to: {self.args.output}")
            print(f"[DONE] Total frames processed: {self.frame_count}")
            total_tiles = self.args.tiles_x * self.args.tiles_y
            print(f"[DONE] Tiles per frame: {total_tiles}\n")
        
        return 0


def main():
    try:
        args = parse_args()
        pipeline = TilingResolutionPreservingPipeline(args)
        return pipeline.run()
    
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease check the parameters and try again.\n")
        return 1
    
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())