import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from src.safety_shield import SafetyShieldWrapper
import time


def test_shield_mechanics():
    print("🔧 INICIANDO DIAGNÓSTICO DEL ESCUDO...")

    # 1. Configuración idéntica al entrenamiento
    env_config = {
        "use_render": True,  # IMPORTANTE: True para que VEAS qué pasa
        "manual_control": False,
        "traffic_density": 0.0,  # Sin tráfico para aislar el problema del carril
        "map": "SSSSSS",
        "vehicle_config": {
            "lidar": {"num_lasers": 240, "distance": 50, "num_others": 4},
        },
    }

    # 2. Inicializar entorno y envolverlo
    env = MetaDriveEnv(env_config)
    env = SafetyShieldWrapper(
        env, lidar_threshold=0.25, num_lasers=240, lane_threshold=0.8
    )

    obs, info = env.reset()
    print("✅ Entorno Reseteado. Iniciando 'Agente Suicida'...")
    print("   -> El agente intentará girar SIEMPRE a la derecha (Acción: [1.0, 0.5])")
    print("   -> El escudo DEBERÍA corregirlo hacia la izquierda.")

    total_steps = 0
    shield_interventions = 0

    try:
        for i in range(200):  # Intentar correr 200 pasos
            # ACCION SUICIDA: Girar todo a la derecha (1.0) y acelerar (0.5)
            # En MetaDrive: Steering +1.0 es Izquierda o Derecha dependiendo de config,
            # probaremos girar fuerte.
            suicide_action = np.array([-1.0, 0.5])

            # Paso del entorno
            obs, reward, done, truncated, info = env.step(suicide_action)
            env.render()

            # Verificación de datos
            lat_dist = "N/A"
            vehicle = env.unwrapped.agent
            if vehicle and vehicle.lane:
                _, lat = vehicle.lane.local_coordinates(vehicle.position)
                lat_dist = round(lat, 2)

            is_shield_active = info.get("shield_activated", False)
            if is_shield_active:
                shield_interventions += 1
                status = "🛡️ SALVADO"
            else:
                status = "⚠️ PELIGRO"

            print(
                f"Step {i} | LatDist: {lat_dist} | Shield: {is_shield_active} | {status}"
            )

            total_steps += 1
            time.sleep(0.05)  # Para que te de tiempo a verlo

            if done:
                print(f"❌ CHOQUE en el paso {i}. Causa: {info}")
                break

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("\n--- REPORTE DE DIAGNÓSTICO ---")
        print(f"Pasos totales: {total_steps}")
        print(f"Intervenciones del escudo: {shield_interventions}")
        if total_steps < 10:
            print("🔴 FALLO CRÍTICO: El escudo no evitó la muerte inmediata.")
        elif shield_interventions == 0:
            print(
                "🔴 FALLO DE LÓGICA: El coche sobrevivió (o no) pero el escudo nunca saltó."
            )
        else:
            print("🟢 ÉXITO: El escudo intervino activamente.")


if __name__ == "__main__":
    test_shield_mechanics()
