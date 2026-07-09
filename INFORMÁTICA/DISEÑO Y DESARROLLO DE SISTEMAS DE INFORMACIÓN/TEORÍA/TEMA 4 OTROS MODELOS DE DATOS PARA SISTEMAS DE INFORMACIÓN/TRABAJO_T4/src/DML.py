import transaction
from DDL import Cliente, Trabajador, Reserva
import datetime


def crear_cliente(root, cliente: Cliente):
    clientes = root.get('clientes', {})
    if cliente.dni in clientes:
        raise KeyError(f"Cliente {cliente.dni} ya existe")
    clientes[cliente.dni] = cliente


def crear_trabajador(root, trabajador: Trabajador):
    trabajadores = root.get('trabajadores', {})
    if trabajador.dni in trabajadores:
        raise KeyError(f"Trabajador {trabajador.dni} ya existe")
    trabajadores[trabajador.dni] = trabajador


def crear_reserva(root, reserva: Reserva):
    reservas = root.get('reservas', {})
    if reserva.id_reserva in reservas:
        raise KeyError(f"Reserva {reserva.id_reserva} ya existe")
    reservas[reserva.id_reserva] = reserva

def insertar_datos_ejemplo(root):
    # Crear objetos (DML - Inserción)
    cli = Cliente("12345678A", "Leandro", "Fernández", "600111222", "lea@email.com", "1234")
    tra = Trabajador("87654321B", "Ana", "López", "600333444", "ana@rest.com", "Cocinero")

    # Persistir objetos con comprobación de clave primaria
    try:
        crear_cliente(root, cli)
    except KeyError as e:
        print(f"{e}\n")

    try:
        crear_trabajador(root, tra)
    except KeyError as e:
        print(f"{e}\n")

    # Crear reserva usando los objetos anteriores (Objeto complejo)
    res = Reserva("RES-001", cli, tra, 4, datetime.datetime(2025, 12, 31, 21, 0), "Confirmada")
    try:
        crear_reserva(root, res)
    except KeyError as e:
        print(f"{e}\n")

    transaction.commit() # Garantiza la Durabilidad (ACID)
    print("Datos insertados correctamente.\n")


def delete_reserva(root, id_reserva: str):
    reservas = root.get('reservas', {})
    if id_reserva not in reservas:
        raise KeyError(f"Reserva {id_reserva} no existe")

    del reservas[id_reserva]
    transaction.commit()


def update_reserva_reprogramar(root, id_reserva: str, nueva_fecha_hora: datetime.datetime, nuevo_num_personas: int | None = None):
    reservas = root.get('reservas', {})
    if id_reserva not in reservas:
        raise KeyError(f"Reserva {id_reserva} no existe")

    res = reservas[id_reserva]
    res.fecha_hora = nueva_fecha_hora
    if nuevo_num_personas is not None:
        res.num_personas = nuevo_num_personas

    res._p_changed = True
    transaction.commit()
    return res

def update_reserva_reasignar_trabajador(root, id_reserva: str, dni_trabajador: str):
    reservas = root.get('reservas', {})
    trabajadores = root.get('trabajadores', {})
    if id_reserva not in reservas:
        raise KeyError(f"Reserva {id_reserva} no existe")
    if dni_trabajador not in trabajadores:
        raise KeyError(f"Trabajador {dni_trabajador} no existe")

    res = reservas[id_reserva]
    res.trabajador = trabajadores[dni_trabajador]  # referencia directa a objeto
    res._p_changed = True
    transaction.commit()
    return res

