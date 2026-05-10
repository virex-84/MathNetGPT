//https://github.com/virex-84

using MathNet.Numerics.LinearAlgebra;
using MessagePack;

namespace Llm.Core;

[MessagePackObject]
public class SerializableMatrix
{
    [Key(0)] public int Rows { get; set; }
    [Key(1)] public int Cols { get; set; }
    [Key(2)] public float[] Data { get; set; }

    public SerializableMatrix(int rows, int cols, float[] data)
    {
        Rows = rows;
        Cols = cols;
        Data = data;
    }

    public SerializableMatrix()
    {
        Rows = 0;
        Cols = 0;
        Data = Array.Empty<float>();
    }

    public static SerializableMatrix FromMatrix(Matrix<float> matrix)
    {
        return new SerializableMatrix(
            matrix.RowCount,
            matrix.ColumnCount,
            matrix.ToRowMajorArray());
    }

    public Matrix<float> ToMatrix()
    {
        // FIX: используем DenseOfRowMajor — MathNet внутри
        // корректно конвертирует row-major → column-major хранение
        // Это быстрее чем цикл через индексатор [i,j]
        return Matrix<float>.Build.DenseOfRowMajor(Rows, Cols, Data);
    }
}