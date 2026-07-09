#from db.queries import mostrar_tablas, nuevo_pedido, crear_tablas_y_datos
#from db.transactions import add_detalle

# Arregla error ImportError 
from db.queries import mostrar_tablas, crear_tablas_y_datos
from db.transactions import add_detalle, nuevo_pedido


def main_menu(conn, cursor):
    """
    Muestra el menú principal del sistema de gestión de pedidos.

    Permite al usuario:
      - Crear las tablas e insertar datos iniciales.
      - Dar de alta un nuevo pedido.
      - Visualizar el contenido de las tablas.
      - Salir del programa.

    Args:
        conn (oracledb.Connection): 
            Conexión activa con la base de datos Oracle.
        cursor (oracledb.Cursor): 
            Cursor utilizado para ejecutar las operaciones SQL.

    Returns:
        None
    """
    while True:
        print("\n--- MENÚ PRINCIPAL ---")
        print("1. Borrado y nueva creación de tablas Stock")
        print("2. Dar de alta nuevo pedido")
        print("3. Mostrar tablas")
        print("4. Salir")
        op = input("Opción: ")
    
        if op == "1":
            #crea tablas y carga datos iniciales
            crear_tablas_y_datos(cursor)
        if op == "2":
            cpedido_nuevo = nuevo_pedido(conn, cursor) #Llama a nuevo_pedido y guarda ID que devuelve
            submenu_pedido(conn, cursor, cpedido_nuevo) #Ahora llama al submenú con ese ID
        elif op == "3":
            #se muestra el contenido de las tablas
            mostrar_tablas(cursor)
        elif op == "4":
            #sale del programa
            print("Cerrando conexión...")
            break
        else:
            print("Opción no válida.")


def submenu_pedido(connection, cursor, cpedido):
    """
    Submenú que permite gestionar las operaciones relacionadas con un pedido.

    Ofrece opciones para:
      1. Añadir detalles de productos.
      2. Eliminar los detalles añadidos.
      3. Cancelar completamente el pedido (rollback).
      4. Confirmar la transacción (commit).

    Args:
        connection (oracledb.Connection): 
            Conexión activa con la base de datos Oracle.
        cursor (oracledb.Cursor): 
            Cursor para ejecutar sentencias SQL.
        cpedido (int): 
            Código del pedido actualmente en curso.

    Returns:
        None
    """
    while True:
        print("\n--- DAR DE ALTA NUEVO PEDIDO ---")
        print("1. Añadir detalle de producto")
        print("2. Eliminar todos los detalles")
        print("3. Cancelar pedido (ROLLBACK)")
        print("4. Finalizar pedido (COMMIT)")
        opcion = input("Elige opción: ")

        if opcion == "1":
            #añadir un detalle en caso de haber stock suficiente
            add_detalle(connection, cursor, cpedido)
            mostrar_tablas(cursor)
        elif opcion == "2":
            cursor.execute("ROLLBACK TO SAVEPOINT pedido_creado") #revierte cambios -> pero no borra el pedido
            print("Detalles eliminados.")
            mostrar_tablas(cursor)
        elif opcion == "3":
            cursor.execute("ROLLBACK TO SAVEPOINT antes_pedido") #cancela todo el pedido -> aquí si se borra
            print("Pedido cancelado.")
            mostrar_tablas(cursor)
            break
        elif opcion == "4":
            connection.commit() #solo dejamos este commit pues es el que confirma el final de la transaccion
            print("Pedido confirmado (COMMIT).")
            break
        else:
            print("Opción inválida.")
