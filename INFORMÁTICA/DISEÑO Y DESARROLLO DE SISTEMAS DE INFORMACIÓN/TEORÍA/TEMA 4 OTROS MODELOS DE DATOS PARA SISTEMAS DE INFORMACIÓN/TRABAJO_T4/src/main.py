import datetime
import ZODB, ZODB.FileStorage
import DDL
import DML
import queries


def main():
    
    # Abrir conexión ZODB y obtener root
    storage = ZODB.FileStorage.FileStorage('../db/restaurante.fs')
    db = ZODB.DB(storage)
    connection = db.open()
    root = connection.root()

    # Crear Tablas
    try:
        DDL.create_tables(root)
        print("Tablas creadas correctamente.\n")
    except Exception as e:
        print(f"create_tables: {e}\n")

    # Insertar datos de ejemplo
    try:
        DML.insertar_datos_ejemplo(root)
    except Exception as e:
        print(f"insertar_datos_ejemplo: {e}\n")

    # Listar reservas detalladas
    try:
        queries.listar_reservas_detalladas(root)
    except Exception as e:
        print(f"listar_reservas_detalladas: {e}\n")

    # Consultar reservas por cliente
    try:
        dni = "12345678A"
        res_cli = queries.reservas_por_cliente(root, dni)
        print(f"Reservas del cliente {dni}: {res_cli}\n")
    except Exception as e:
        print(f"reservas_por_cliente: {e}\n")

    # Conteo de reservas por día
    try:
        conteo = queries.conteo_reservas_por_dia(root)
        print(f"Conteo reservas por día: {conteo}\n")
    except Exception as e:
        print(f"conteo_reservas_por_dia: {e}\n")

    # Reprogramar reserva
    try:
        nueva_fecha = datetime.datetime(2026, 1, 1, 20, 0)
        res = DML.update_reserva_reprogramar(root, "RES-001", nueva_fecha, nuevo_num_personas=2)
        print(f"Reserva reprogramada: id={getattr(res, 'id_reserva', None)}, fecha={getattr(res, 'fecha_hora', None)}, personas={getattr(res, 'num_personas', None)}\n")
    except Exception as e:
        print(f"update_reserva_reprogramar: {e}\n")

    # Reasignar trabajador a reserva
    try:
        res = DML.update_reserva_reasignar_trabajador(root, "RES-001", "87654321B")
        trabajador = getattr(res, "trabajador", None)
        print(f"Reserva reasignada: id={getattr(res, 'id_reserva', None)}, trabajador={getattr(trabajador, 'dni', None) if trabajador else None}\n")
    except Exception as e:
        print(f"update_reserva_reasignar_trabajador: {e}\n")

    # Listado final
    try:
        queries.listar_reservas_detalladas(root)
    except Exception as e:
        print(f"listar_reservas_detalladas (final): {e}\n")

    # Borrar reserva
    try:
        DML.delete_reserva(root, "RES-001")
        print("Reserva RES-001 eliminada.\n")
    except Exception as e:
        print(f"delete_reserva: {e}\n")

    # Listado después de borrado
    try:
        queries.listar_reservas_detalladas(root)
    except Exception as e:
        print(f"listar_reservas_detalladas (post-delete): {e}\n")

    # Eliminar Tablas
    try:
        DDL.drop_tables(root)
        print("Tablas eliminadas correctamente.\n")
    except Exception as e:
        print(f"drop_tables: {e}\n")
        
    finally:
        try:
            connection.close()
            db.close()
            storage.close()
            print ("Conexión cerrada correctamente.\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()