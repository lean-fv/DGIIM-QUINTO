from persistent import Persistent
import transaction


class Persona(Persistent):
    def __init__(self, dni, nombre, apellidos, telefono, email):
        self.dni = dni
        self.nombre = nombre
        self.apellidos = apellidos
        self.telefono = telefono
        self.email = email

class Cliente(Persona): # Herencia
    def __init__(self, dni, nombre, apellidos, telefono, email, contrasena, estado_sesion="Desconectado"):
        super().__init__(dni, nombre, apellidos, telefono, email)
        self.contrasena = contrasena
        self.estado_sesion = estado_sesion

class Trabajador(Persona): # Herencia
    def __init__(self, dni, nombre, apellidos, telefono, email, cargo, estado="Activo"):
        super().__init__(dni, nombre, apellidos, telefono, email)
        self.cargo = cargo
        self.estado = estado

class Reserva(Persistent): # Objeto Complejo
    def __init__(self, id_reserva, cliente, trabajador, num_personas, fecha_hora, estado="Pendiente"):
        self.id_reserva = id_reserva
        self.cliente = cliente # Referencia directa al objeto Cliente
        self.trabajador = trabajador # Referencia directa al objeto Trabajador
        self.num_personas = num_personas
        self.fecha_hora = fecha_hora
        self.estado = estado


def create_tables(root):
    if 'clientes' not in root:
        root['clientes'] = {}
    if 'trabajadores' not in root:
        root['trabajadores'] = {}
    if 'reservas' not in root:
        root['reservas'] = {}
    root ._p_changed = True
    transaction.commit()

def drop_tables(root):
    for key in ['clientes', 'trabajadores', 'reservas']:
        if key in root:
            del root[key]
    root._p_changed = True
    transaction.commit()
