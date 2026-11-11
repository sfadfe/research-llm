import re
import struct
import sys

TYPES = {0: "vendor", 1: "invalid", 2: "dispatch", 3: "barrier_and", 4: "agent", 5: "barrier_or"}


def parse(path):
    node = None
    cur = None
    out = []
    for line in open(path, errors="replace"):
        if line.startswith("Node"):
            node = line.split()[3].rstrip(":")
            cur = None
            continue
        m = re.match(r"\s*Inst \d+,\s+CP Pipe (\d+), Queue (\d+)", line)
        if m:
            cur = {"name": f"gpu {node} P{m[1]} Q{m[2]}", "skip": 0, "live": True, "d": []}
            out.append(cur)
            continue
        m = re.match(r"\s*Compute queue on device (\S+)", line)
        if m:
            cur = {"name": f"gpu {m[1]} mqd", "skip": 128, "live": False, "d": []}
            out.append(cur)
            continue
        if re.match(r"\s*(Inst|SDMA)|Process", line):
            cur = None
            continue
        m = re.match(r"\s*[0-9a-f]{8}: (.*)", line)
        if m and cur is not None:
            cur["d"] += [int(x, 16) for x in m[1].split()]
    for q in out:
        q["d"] = q["d"][q["skip"]:]
    return [q for q in out if len(q["d"]) > 17 and (q["d"][8] or q["d"][9])]


def read_maps(pid):
    res = []
    for line in open(f"/proc/{pid}/maps"):
        lo, hi = line.split()[0].split("-")
        res.append((int(lo, 16), int(hi, 16)))
    return res


def mapped(maps, addr):
    return any(lo <= addr < hi for lo, hi in maps)


def describe(raw):
    h, setup, wx, wy, wz, _, gx, gy, gz, priv, group, kobj, karg, _, sig = struct.unpack("<HHHHHHIIIIIQQQQ", raw)
    kind = TYPES.get(h & 0xff, f"t{h & 0xff}")
    if kind == "dispatch":
        return f"hdr {h:#06x} {kind} wg {wx}x{wy}x{wz} grid {gx}x{gy}x{gz} priv {priv} group {group} kobj {kobj:#x}"
    return f"hdr {h:#06x} {kind}"


def main():
    queues = parse(sys.argv[1])
    for pid in sys.argv[2:]:
        maps = read_maps(pid)
        with open(f"/proc/{pid}/mem", "rb", buffering=0) as mem:
            def peek(addr, n):
                mem.seek(addr)
                return mem.read(n)

            for q in queues:
                d = q["d"]
                base = (d[9] << 32 | d[8]) << 8
                if not mapped(maps, base):
                    continue
                size = 1 << ((d[17] & 0x3f) + 3)
                report = d[12] << 32 | d[11]
                poll = d[14] << 32 | d[13]
                rep = struct.unpack("<Q", peek(report, 8))[0] if mapped(maps, report) else -1
                wptr = struct.unpack("<Q", peek(poll, 8))[0] if mapped(maps, poll) else -1
                rptr = d[10] * 4 if q["live"] else rep * 64
                print(f"pid {pid} {q['name']} base {base:#x} size {size} hqd_rptr_dw {d[10]:#x} report {rep} wptr {wptr} doorbell {d[15]:#x}")
                if size < 1 << 16:
                    continue
                for i in range(-3, 9):
                    off = (rptr + i * 64) % size
                    print(f"  {i:+d} {off:#08x} {describe(peek(base + off, 64))}")


if __name__ == "__main__":
    main()
