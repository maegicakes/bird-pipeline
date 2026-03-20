# -*- coding: utf-8 -*-
import os
import time
import socket
import subprocess
from datetime import datetime, timezone

# Hardware / I2C
import board
import busio
from adafruit_ina260 import INA260

# DB
import psycopg
from psycopg.rows import tuple_row

DATABASE_URL = os.environ.get("DATABASE_URL")  # e.g. postgres://user:pass@host:5432/db
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var is required")

DEVICE_ID = os.environ.get("DEVICE_ID") or socket.gethostname()

INSERT_SQL = """
INSERT INTO pi_heartbeats (time, station_id, voltage, current, power, soc_pct, cpu_temp)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

# 25°C resting-voltage table for ~12V SLA (rough guide)
v_soc_table = [
    (13.10, 100),
    (12.95,  90),
    (12.85,  80),
    (12.75,  70),
    (12.70,  60),
    (12.50,  50),
    (12.30,  40),
    (12.10,  30),
    (11.95,  20),
    (11.80,  10),
    (11.60,   0),
]

def open_conn():
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=tuple_row)


def insert_heartbeat(conn, when_utc, device_id, voltage_v, current_a, power_w, est_batt_p, cpu_temp_c):
    with conn.cursor() as cur:
        cur.execute(
            INSERT_SQL,
            (when_utc, device_id, voltage_v, current_a, power_w, est_batt_p, cpu_temp_c),
        )


def soc_from_voltage(v):
    for threshold, soc in V_SOC_TABLE:
        if v >= threshold:
            return soc
    return 0


def get_cpu_temp():
    """
    Try vcgencmd (host RPi tool). If unavailable in container, fall back to /sys thermal zone.
    """
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True, timeout=1.0)
        # e.g. "temp=42.8'C"
        return float(out.replace("temp=", "").replace("'C", "").strip())
    except Exception:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                milli = int(f.read().strip())
            return milli / 1000.0
        except Exception:
            return None


def main():
    # init I2C + INA260
    i2c = busio.I2C(board.SCL, board.SDA)
    # default INA260 addr is 0x40; adjust with env if needed
    addr = int(os.environ.get("INA260_ADDR", "0x40"), 16)
    ina = INA260(i2c, address=addr)

    db = None
    backoff = 1.0
    interval_s = int(os.environ.get("HEARTBEAT_INTERVAL_SEC", "600"))
    print(f"[heartbeat] reading INA260 on addr=0x{addr:02X}, interval={interval_s}s. Ctrl+C to stop.")

    while True:
        try:
            if db is None:
                db = open_conn()
                backoff = 1.0

            # INA260 units: V, mA, mW
            voltage_v = float(ina.voltage)
            current_a = float(ina.current) / 1000.0
            power_w = float(ina.power) / 1000.0
            soc_pct = soc_from_voltage(voltage_v)
            cpu_temp_c = get_cpu_temp()

            now_utc = datetime.now(timezone.utc)
            print(
                f"[heartbeat] {now_utc.isoformat()} dev={DEVICE_ID} "
                f"V={voltage_v:.2f}V I={current_a:.3f}A P={power_w:.3f}W "
                f"SOC≈{soc_pct:3d}% CPU={cpu_temp_c if cpu_temp_c is not None else 'NA'}°C"
            )

            insert_heartbeat(db, now_utc, DEVICE_ID, voltage_v, current_a, power_w, soc_pct, cpu_temp_c)
            time.sleep(interval_s)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[heartbeat] {e}; retrying in {backoff:.1f}s")
            try:
                if db is not None:
                    db.close()
            except Exception:
                pass
            db = None
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


if __name__ == "__main__":
    main()
