//https://github.com/virex-84

namespace Llm.Core;

public static class Tokenizer
{
    public static List<int> Tokenize(Vocab vocab, string text)
    {
        var tokens = new List<int>();
        // Используем Span-based split если доступно, иначе обычный Split
        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        var sb = new System.Text.StringBuilder();

        foreach (var word in words)
        {
            if (word == "</s>")
            {
                tokens.Add(vocab.EosTokenId);
                continue;
            }

            sb.Clear();

            foreach (var c in word)
            {
                if (char.IsPunctuation(c))
                {
                    if (sb.Length > 0)
                    {
                        tokens.Add(vocab.EncodeWordSafe(sb.ToString()));
                        sb.Clear();
                    }
                    tokens.Add(vocab.EncodeWordSafe(c.ToString()));
                }
                else
                {
                    sb.Append(c);
                }
            }

            if (sb.Length > 0)
            {
                tokens.Add(vocab.EncodeWordSafe(sb.ToString()));
            }
        }

        return tokens;
    }
}
