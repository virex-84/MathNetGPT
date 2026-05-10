//https://github.com/virex-84

using MessagePack;
using System.Text;

namespace Llm.Core;

[MessagePackObject]
public class Vocab
{
    public const string UnkToken = "<unk>";
    public const string EosToken = "</s>";
    public const string PadToken = "<pad>";

    [Key(0)]
    public Dictionary<string, int> Encode { get; set; }

    [Key(1)]
    public Dictionary<int, string> Decode { get; set; }

    [Key(2)]
    public List<string> Words { get; set; }

    // Кэшируем ID специальных токенов — без lookup в словарь каждый раз
    // IgnoreMember — не сериализуем, восстанавливаем при загрузке
    [IgnoreMember]
    public int UnkTokenId { get; private set; }

    [IgnoreMember]
    public int EosTokenId { get; private set; }

    [IgnoreMember]
    public int PadTokenId { get; private set; }

    public Vocab()
    {
        Encode = new Dictionary<string, int>();
        Decode = new Dictionary<int, string>();
        Words = new List<string>(DefaultWords());

        for (int i = 0; i < Words.Count; i++)
        {
            Encode[Words[i]] = i;
            Decode[i] = Words[i];
        }
        CacheSpecialTokenIds();
    }

    public Vocab(IEnumerable<string> words)
    {
        Encode = new Dictionary<string, int>();
        Decode = new Dictionary<int, string>();

        var wordSet = new HashSet<string>(words);
        wordSet.Add(UnkToken);
        wordSet.Add(EosToken);
        wordSet.Add(PadToken);

        var wordList = wordSet.ToList();
        wordList.Sort();

        for (int i = 0; i < wordList.Count; i++)
        {
            Encode[wordList[i]] = i;
            Decode[i] = wordList[i];
        }
        Words = wordList;
        CacheSpecialTokenIds();
    }

    /// <summary>
    /// Вызывается после десериализации MessagePack для восстановления кэшей
    /// </summary>
    public void PostDeserialize()
    {
        CacheSpecialTokenIds();
    }

    private void CacheSpecialTokenIds()
    {
        UnkTokenId = Encode.TryGetValue(UnkToken, out int unk) ? unk : 0;
        EosTokenId = Encode.TryGetValue(EosToken, out int eos) ? eos : 1;
        PadTokenId = Encode.TryGetValue(PadToken, out int pad) ? pad : 2;
    }

    public int EncodeWordSafe(string word)
    {
        return Encode.TryGetValue(word, out int tokenId) ? tokenId : UnkTokenId;
    }

    public int? EncodeWord(string word)
    {
        return Encode.TryGetValue(word, out int tokenId) ? tokenId : null;
    }

    public string DecodeToken(int tokenId)
    {
        return Decode.TryGetValue(tokenId, out string? word) ? word ?? UnkToken : UnkToken;
    }

    public static IEnumerable<string> DefaultWords()
    {
        return new[] { PadToken, UnkToken, EosToken, "hello", "world", "this", "is", "c#" };
    }

    public static void ProcessTextForVocab(IEnumerable<string> texts, HashSet<string> vocabSet)
    {
        vocabSet.Add(EosToken);
        vocabSet.Add(UnkToken);
        vocabSet.Add(PadToken);

        var sb = new StringBuilder();

        foreach (var text in texts)
        {
            foreach (var word in text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
            {
                if (word == EosToken || word == UnkToken || word == PadToken)
                {
                    vocabSet.Add(word);
                    continue;
                }

                sb.Clear();
                foreach (char c in word)
                {
                    if (char.IsPunctuation(c))
                    {
                        if (sb.Length > 0)
                        {
                            vocabSet.Add(sb.ToString());
                            sb.Clear();
                        }
                        vocabSet.Add(c.ToString());
                    }
                    else
                    {
                        sb.Append(c);
                    }
                }

                if (sb.Length > 0)
                    vocabSet.Add(sb.ToString());
            }
        }
    }
}