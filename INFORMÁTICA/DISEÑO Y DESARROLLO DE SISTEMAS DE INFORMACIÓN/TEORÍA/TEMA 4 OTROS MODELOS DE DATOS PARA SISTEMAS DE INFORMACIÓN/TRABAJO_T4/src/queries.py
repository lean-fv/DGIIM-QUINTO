def listar_reservas_detalladas(root):
    print("--- LISTADO DE RESERVAS ---\n")
    reservas = root.get('reservas', {})
    
    for id_res, res in reservas.items():
        # Navegación natural entre objetos (Sin necesidad de JOINs)
        print(f"Reserva ID: {id_res}\n")
        print(f"  Cliente: {res.cliente.nombre} {res.cliente.apellidos}\n")
        print(f"  Gestionada por: {res.trabajador.nombre} ({res.trabajador.cargo})\n")
        print(f"  Fecha/Hora: {res.fecha_hora}\n")
        print(f"  Estado: {res.estado}\n")
        print(("-" * 30) + "\n")


def reservas_por_cliente(root, dni_cliente: str):
    reservas = root.get('reservas', {})
    return [r for r in reservas.values() if r.cliente.dni == dni_cliente]

def conteo_reservas_por_dia(root):
    reservas = root.get('reservas', {})
    counts = {}
    for r in reservas.values():
        d = r.fecha_hora.date()
        counts[d] = counts.get(d, 0) + 1
    return counts
