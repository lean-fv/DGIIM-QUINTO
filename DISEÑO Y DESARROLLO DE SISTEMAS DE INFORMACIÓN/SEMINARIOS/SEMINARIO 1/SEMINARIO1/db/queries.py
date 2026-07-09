import datetime

def crear_tablas_y_datos(cursor):
    """
    Elimina las tablas existentes (si las hay) y crea de nuevo
    las tablas 'Stock', 'Pedido' y 'Detalle_Pedido'.

    Además, inserta un conjunto de registros iniciales en la tabla 'Stock'.

    Args:
        cursor (oracledb.Cursor): 
            Cursor para ejecutar las sentencias SQL.

    Returns:
        None
    """
    cursor.execute("BEGIN EXECUTE IMMEDIATE 'DROP TABLE Detalle_Pedido'; EXCEPTION WHEN OTHERS THEN NULL; END;")
    cursor.execute("BEGIN EXECUTE IMMEDIATE 'DROP TABLE Pedido'; EXCEPTION WHEN OTHERS THEN NULL; END;")
    cursor.execute("BEGIN EXECUTE IMMEDIATE 'DROP TABLE Stock'; EXCEPTION WHEN OTHERS THEN NULL; END;")
    
    cursor.execute("""
        CREATE TABLE Stock (
            Cproducto NUMBER PRIMARY KEY,
            Descripcion VARCHAR2(100),
            Cantidad NUMBER
        )
    """)
    
    cursor.execute("""
        CREATE TABLE Pedido (
            Cpedido NUMBER PRIMARY KEY,
            Ccliente NUMBER,
            Fecha DATE
        )
    """)
    
    cursor.execute("""
        CREATE TABLE Detalle_Pedido (
            Cpedido NUMBER,
            Cproducto NUMBER,
            Cantidad NUMBER,
            PRIMARY KEY (Cpedido, Cproducto),
            FOREIGN KEY (Cpedido) REFERENCES Pedido(Cpedido),
            FOREIGN KEY (Cproducto) REFERENCES Stock(Cproducto)
        )
    """)
    
    productos = [
        (1, 'Producto A', 100),
        (2, 'Producto B', 150),
        (3, 'Producto C', 200),
        (4, 'Producto D', 250),
        (5, 'Producto E', 300),
        (6, 'Producto F', 350),
        (7, 'Producto G', 400),
        (8, 'Producto H', 450),
        (9, 'Producto I', 500),
        (10, 'Producto J', 550)
    ]
    
    for prod in productos:
        cursor.execute("INSERT INTO Stock VALUES (:1, :2, :3)", prod)
    cursor.connection.commit()
    print("Tablas creadas e insertados datos iniciales en Stock.")

def mostrar_tablas(cursor):
    """
    Muestra por consola el contenido de las tablas principales:
    'Stock', 'Pedido' y 'Detalle_Pedido'.

    Args:
        cursor (oracledb.Cursor): 
            Cursor de base de datos activo.

    Returns:
        None
    """
    tablas = ["Stock", "Pedido", "Detalle_Pedido"]
    for t in tablas:
        print(f"\n--- {t} ---")
        cursor.execute(f"SELECT * FROM {t}")
        for fila in cursor.fetchall():
            print(fila)
