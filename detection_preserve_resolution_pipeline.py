#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resolution-Preserving Hailo Detection Pipeline
----------------------------------------------
Based on tee + hailomuxer architecture to maintain original resolution.
Supports MP4 input/output with configurable model and postprocessing.
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

# Inference resolution (model input size)
INFERENCE_WIDTH = 640
INFERENCE_HEIGHT = 640

# NMS parameters
NMS_SCORE_THRESHOLD = 0.3
NMS_IOU_THRESHOLD = 0.45

# Encoding parameters
BITRATE = 4000  # kbps
ENCODING_PRESET = "ultrafast"  # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow

# ============================================================================

Gst.init(None)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resolution-preserving Hailo detection pipeline for MP4 files"
    )
    parser.add_argument("--input", required=True, help="Input MP4 file path")
    parser.add_argument("--output", required=True, help="Output MP4 file path")
    parser.add_argument("--hef", default=DEFAULT_HEF_PATH, help="HEF model file path")
    parser.add_argument("--postprocess-so", default=DEFAULT_POSTPROCESS_SO, 
                       help="Postprocess .so library path")

    parser.add_argument("--inference-width", type=int, default=INFERENCE_WIDTH,
                       help="Inference width (model input)")
    parser.add_argument("--inference-height", type=int, default=INFERENCE_HEIGHT,
                       help="Inference height (model input)")
    parser.add_argument("--bitrate", type=int, default=BITRATE,
                       help="Output video bitrate in kbps")
    parser.add_argument("--debug", action="store_true",
                       help="Print debug information")
    return parser.parse_args()


class ResolutionPreservingPipeline:
    def __init__(self, args):
        self.args = args
        self.pipeline = None
        self.loop = None
        self.frame_count = 0
        
        # Validate paths
        self._validate_paths()
        
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
    
    def build_pipeline(self):
        """
        Build GStreamer pipeline with resolution preservation.
        
        Pipeline structure:
        1. Source (filesrc) -> decode
        2. Tee splits to:
           a) Original resolution path -> muxer
           b) Inference path -> rescale -> hailonet -> hailofilter -> muxer
        3. Muxer combines both -> hailooverlay -> encode -> save
        """
        
        nms_params = (
            f'nms-score-threshold={NMS_SCORE_THRESHOLD} '
            f'nms-iou-threshold={NMS_IOU_THRESHOLD} '
            'output-format-type=HAILO_FORMAT_TYPE_FLOAT32'
        )
        
        pipeline_str = f"""
            filesrc location="{self.args.input}" !
            qtdemux !
            h264parse !
            avdec_h264 !
            videoconvert !
            tee name=t
            
            hailomuxer name=mux
            
            t. ! 
            queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
            mux.
            
            t. !
            videoscale !
            video/x-raw,width={self.args.inference_width},height={self.args.inference_height} !
            queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
            hailonet hef-path={self.args.hef} is-active=true {nms_params} !
            queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
            hailofilter so-path={self.args.postprocess_so} qos=false !
            mux.
            
            mux. !
            queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
            hailooverlay !
            queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !
            videoconvert !
            x264enc bitrate={self.args.bitrate} speed-preset={ENCODING_PRESET} tune=zerolatency key-int-max=60 !
            h264parse !
            mp4mux !
            filesink location="{self.args.output}" sync=false
        """
        
        # Clean up the string (remove extra whitespace)
        pipeline_str = " ".join(pipeline_str.split())
        
        if self.args.debug:
            print("\n[DEBUG] GStreamer Pipeline:")
            print("-" * 80)
            # Print formatted for readability
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
        """Optional: Monitor processing (can be removed if not needed)"""
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            print(f"[PROGRESS] Processed {self.frame_count} frames...", end='\r')
        
        return Gst.PadProbeReturn.OK
    
    def run(self):
        """Build and run the pipeline"""
        print("\n" + "=" * 80)
        print("Resolution-Preserving Hailo Detection Pipeline")
        print("=" * 80)
        print(f"Input:            {self.args.input}")
        print(f"Output:           {self.args.output}")
        print(f"HEF:              {self.args.hef}")
        print(f"Postprocess:      {self.args.postprocess_so}")
        print(f"Inference size:   {self.args.inference_width}x{self.args.inference_height}")
        print(f"Output bitrate:   {self.args.bitrate} kbps")
        print("=" * 80)
        print()
        
        # Build pipeline
        try:
            pipeline_str = self.build_pipeline()
            self.pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            print(f"[ERROR] Failed to create pipeline: {e}")
            return 1
        
        # Optional: Add buffer probe to monitor progress
        # You can remove this if not needed
        mux = self.pipeline.get_by_name("mux")
        if mux:
            src_pad = mux.get_static_pad("src")
            if src_pad:
                src_pad.add_probe(Gst.PadProbeType.BUFFER, self.on_buffer_probe)
        
        # Setup message bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)
        
        # Create main loop
        self.loop = GLib.MainLoop()
        
        # Start pipeline
        print("[START] Starting pipeline...\n")
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
            print(f"[DONE] Total frames processed: {self.frame_count}\n")
        
        return 0


def main():
    try:
        args = parse_args()
        pipeline = ResolutionPreservingPipeline(args)
        return pipeline.run()
    
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease check the file paths and try again.\n")
        return 1
    
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())