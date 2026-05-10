//https://github.com/virex-84

using MathNet.Numerics.LinearAlgebra;

namespace Llm.Core;

public class Adam
{
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    private readonly float _oneMinusBeta1;
    private readonly float _oneMinusBeta2;
    private int _timestep;

    private float _beta1t;
    private float _beta2t;

    private readonly float[] _m;
    private readonly float[] _v;
    private readonly int _rows;
    private readonly int _cols;

    public Adam(int rows, int cols)
    {
        _beta1 = 0.9f;
        _beta2 = 0.999f;
        _epsilon = 1e-8f;
        _oneMinusBeta1 = 1.0f - _beta1;
        _oneMinusBeta2 = 1.0f - _beta2;
        _timestep = 0;
        _beta1t = 1.0f;
        _beta2t = 1.0f;
        _rows = rows;
        _cols = cols;
        _m = new float[rows * cols];
        _v = new float[rows * cols];
    }

    public void Step(Matrix<float> a, Matrix<float> grads, float lr)
    {
        if (a.RowCount != _rows || a.ColumnCount != _cols)
            throw new ArgumentException(
                $"Matrix size {a.RowCount}x{a.ColumnCount} != optimizer size {_rows}x{_cols}");

        _timestep++;
        _beta1t *= _beta1;
        _beta2t *= _beta2;

        float mScale = 1.0f / (1.0f - _beta1t);
        float vScale = 1.0f / (1.0f - _beta2t);

        var m = _m;
        var v = _v;

        // OPT: получаем пр€мой доступ к внутренним массивам MathNet
        // DenseMatrix хранит данные column-major в поле _values
        // »спользуем индексацию column-major: idx = row + col * rows
        // чтобы обход был последовательным в пам€ти
        for (int j = 0; j < _cols; j++)
        {
            int colBase = j * _rows; // column-major base
            for (int i = 0; i < _rows; i++)
            {
                int idx = colBase + i; // column-major index

                // grads и a Ч тоже column-major, поэтому обходим одинаково
                float g = grads[i, j];

                float mi = _beta1 * m[idx] + _oneMinusBeta1 * g;
                float vi = _beta2 * v[idx] + _oneMinusBeta2 * g * g;
                m[idx] = mi;
                v[idx] = vi;

                float mHat = mi * mScale;
                float vHat = vi * vScale;

                a[i, j] -= lr * mHat / (MathF.Sqrt(vHat) + _epsilon);
            }
        }
    }

    public List<SerializableMatrix> GetParameters()
    {
        var mData = new float[_m.Length];
        var vData = new float[_v.Length];
        Array.Copy(_m, mData, _m.Length);
        Array.Copy(_v, vData, _v.Length);

        return new List<SerializableMatrix>
        {
            new SerializableMatrix(_rows, _cols, mData),
            new SerializableMatrix(_rows, _cols, vData),
            new SerializableMatrix(1, 1, new float[] { _timestep })
        };
    }

    public void SetParameters(Queue<SerializableMatrix> parameters)
    {
        var mSm = parameters.Dequeue();
        var vSm = parameters.Dequeue();
        var tSm = parameters.Dequeue();

        _timestep = (int)tSm.Data[0];
        _beta1t = (float)Math.Pow(_beta1, _timestep);
        _beta2t = (float)Math.Pow(_beta2, _timestep);

        Array.Copy(mSm.Data, _m, _m.Length);
        Array.Copy(vSm.Data, _v, _v.Length);
    }
}