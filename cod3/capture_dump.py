"""RenderDoc script: dump every indexed draw of a Xenia capture of Call of Duty 3 for capture_check.py.

  qrenderdoc.exe --python cod3/capture_dump.py     (paths from the environment, see below)
CAPTURE_IN = the .rdc file, CAPTURE_OUT = the dump file. Runs inside RenderDoc's own Python (no numpy).
Per draw with at least 30 indices: u32 event id, u32 index count n, u32 bytes per index, then n guest
vertex indices (the title's own index buffer, byte-swapped from big-endian: they index the same vertex
order our extraction reads), then n x 4 floats - each corner's position after the vertex shader (clip
space). Xenia draws the guest's vertex and index buffers as they are, so the shape the game drew can be
compared with ours vertex for vertex.
"""
import os
import struct
import traceback

import renderdoc as rd

src, dst = os.environ["CAPTURE_IN"], os.environ["CAPTURE_OUT"]
log = open(dst + ".log", "w")
try:
    cap = rd.OpenCaptureFile()
    cap.OpenFile(src, "", None)
    res, ctrl = cap.OpenCapture(rd.ReplayOptions(), None)
    draws = []

    def collect(acts):
        for a in acts:
            if a.flags & rd.ActionFlags.Drawcall and a.flags & rd.ActionFlags.Indexed and a.numIndices >= 30:
                draws.append(a)
            collect(a.children)
    collect(ctrl.GetRootActions())
    written = 0
    with open(dst, "wb") as out:
        for d in draws:
            try:
                ctrl.SetFrameEvent(d.eventId, True)
                pipe = ctrl.GetPipelineState()
                if pipe.GetPrimitiveTopology() != rd.Topology.TriangleList:
                    continue
                ib = pipe.GetIBuffer()
                bpi = ib.byteStride or 2
                raw = ctrl.GetBufferData(ib.resourceId, ib.byteOffset + d.indexOffset * bpi, d.numIndices * bpi)
                if bpi == 2:
                    guest = [((v & 0xFF) << 8) | (v >> 8) for v in struct.unpack("<%dH" % d.numIndices, raw)]
                else:
                    guest = list(struct.unpack(">%dI" % d.numIndices, raw))
                mf = ctrl.GetPostVSData(0, 0, rd.MeshDataStage.VSOut)
                if mf.vertexResourceId == rd.ResourceId.Null() or mf.numIndices != d.numIndices:
                    continue
                if mf.indexResourceId != rd.ResourceId.Null():
                    ib2 = ctrl.GetBufferData(mf.indexResourceId, mf.indexByteOffset, mf.numIndices * mf.indexByteStride)
                    vi = struct.unpack("<%d%s" % (mf.numIndices, "H" if mf.indexByteStride == 2 else "I"), ib2)
                else:
                    vi = list(range(mf.numIndices))
                nv = max(vi) + 1
                vb = ctrl.GetBufferData(mf.vertexResourceId, mf.vertexByteOffset, nv * mf.vertexByteStride)
                pos = []
                for k in vi:
                    pos.extend(struct.unpack_from("<4f", vb, k * mf.vertexByteStride))
                out.write(struct.pack("<3I", d.eventId, d.numIndices, bpi))
                out.write(struct.pack("<%dI" % d.numIndices, *guest))
                out.write(struct.pack("<%df" % (4 * d.numIndices), *pos))
                written += 1
            except Exception:
                log.write("draw %d: %s\n" % (d.eventId, traceback.format_exc().splitlines()[-1]))
    log.write("%d indexed draws, %d written\n" % (len(draws), written))
    ctrl.Shutdown()
    cap.Shutdown()
except Exception:
    log.write(traceback.format_exc())
log.close()
os._exit(0)
