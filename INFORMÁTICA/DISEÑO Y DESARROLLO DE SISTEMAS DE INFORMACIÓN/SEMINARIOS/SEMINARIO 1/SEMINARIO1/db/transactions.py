import datetime

# db/transactions.py

import datetime
# demasiado enlace, por eso no ejecuta bien: from ui.menu import submenu_pedido

def nuevo_pedido(connection, cursor):
    """
    Crea un nuevo pedido e inserta su registro en la tabla 'Pedido'.
    Define puntos de guardado (savepoints) para controlar transacciones.

    Args:
        connection (oracledb.Connection): 
            Conexión activa con la base de datos.
        cursor (oracledb.Cursor): 
            Cursor asociado para ejecutar comandos SQL.

    Returns:
        int: Código del pedido creado.
    """
    cursor.execute("SAVEPOINT antes_pedido") #este savepoint nos sirve para la opción 3
    cpedido = int(input("Código de pedido: "))
    ccliente = int(input("Código de cliente: "))

    cursor.execute("INSERT INTO Pedido VALUES (:1, :2, :3)",
                   (cpedido, ccliente, datetime.date.today()))
    #lo quito para crear savepoints en su lugar ->#connection.commit()  # Confirmamos la inserción inicial del pedido
    cursor.execute("SAVEPOINT pedido_creado") #este savepoint nos sirve para la opción 2 -> se borran detalles pero NO el pedido
    print(f"Pedido {cpedido} creado.")

    # quito esta línea que me da error por tanto enlace: submenu_pedido(connection, cursor, cpedido)
    return cpedido # Añadimos



def add_detalle(connection, cursor, cpedido):
    """
    Añade un detalle (producto y cantidad) al pedido indicado.
    Actualiza el stock si hay disponibilidad suficiente.

    Si no hay stock suficiente, revierte la operación mediante un rollback.

    Args:
        connection (oracledb.Connection): 
            Conexión activa con la base de datos.
        cursor (oracledb.Cursor): 
            Cursor activo para ejecutar consultas SQL.
        cpedido (int): 
            Código del pedido al que se añade el detalle.

    Returns:
        None
    """
    cursor.execute("SAVEPOINT detalle_creado")  # Savepoint antes de intentar añadir el detalle
    cproducto = int(input("Código producto: "))
    cantidad = int(input("Cantidad: "))

    cursor.execute("SELECT Cantidad FROM Stock WHERE Cproducto=:1", (cproducto,))
    stock = cursor.fetchone()

    if stock and stock[0] >= cantidad:
        cursor.execute("INSERT INTO Detalle_Pedido VALUES (:1, :2, :3)",
                       (cpedido, cproducto, cantidad))
        cursor.execute("UPDATE Stock SET Cantidad = Cantidad - :1 WHERE Cproducto = :2",
                       (cantidad, cproducto))
        #lo quito para usar savepoints -> connection.commit()
        #cursor.execute("SAVEPOINT detalle_creado")
        print("Detalle añadido y stock actualizado.")
    else:
        cursor.execute("ROLLBACK TO SAVEPOINT detalle_creado")
        print("Stock insuficiente, operación revertida (ROLLBACK).")


