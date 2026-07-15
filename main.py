import time
import sys
import re
import chromadb
from foundry_local_sdk import FoundryLocalManager, Configuration

def clean_output(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    cleaned = []
    for i, line in enumerate(lines):
        if i > 1 and line == lines[i-1] and line == lines[i-2]:
            break 
        cleaned.append(line)
    
    text = "\n".join(cleaned)
    text = re.sub(r'(\n0\s*)+', '\n', text)
    text = re.sub(r'www\.[^\s]+', '', text)
    text = re.sub(r'Notlarda bulunan:.*', '', text, flags=re.IGNORECASE)
    
    if text and text[-1] not in ['.', '!', '?']:
        last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_punct != -1:
            text = text[:last_punct+1]
            
    return text.strip()

def main():
    print("🚀 Sistem başlatılıyor...")
    config = Configuration(app_name="my-first-local-rag-app")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    manager.download_and_register_eps(["CUDAExecutionProvider"])
    
    catalog = manager.catalog
    
    # MODEL YÜKLEME (Hatayı önlemek için parçalı yükleme yöntemi)
    embed_model = catalog.get_model("qwen3-embedding-0.6b")
    embed_model.load()
    embed_client = embed_model.get_embedding_client()
    
    chat_model = catalog.get_model("phi-3.5-mini")
    chat_model.load() 
    chat_client = chat_model.get_chat_client()
    
    # AYARLAR
    chat_client.settings.temperature = 0.0 
    chat_client.settings.max_tokens = 1000 
    
    db_client = chromadb.PersistentClient(path="./vektor_db")
    try:
        collection = db_client.get_collection(name="pdf_notlarim")
    except:
        sys.exit("❌ Veritabanı bulunamadı!")
    
    print("\n✅ SİSTEM HAZIR!")
    
    while True:
        soru = input("\nSen: ")
        if soru.lower() in ['q', 'çıkış', 'exit']: break
        
        try:
            soru_vektoru = embed_client.generate_embeddings(inputs=[soru]).data[0].embedding
            sonuclar = collection.query(query_embeddings=[soru_vektoru], n_results=3, include=['documents', 'distances'])
            
            # Konu dışıysa engelle
            if not sonuclar['documents'] or not sonuclar['documents'][0] or sonuclar['distances'][0][0] > 1.0:
                print("\n🤖 Yapay Zeka: Bu bilgi notlarda bulunmuyor.")
                continue
                
            baglam = "\n\n---\n\n".join(sonuclar['documents'][0])
            sistem_mesaji = (
                f"Bağlam: {baglam}\n\n"
                "KURALLAR:\n"
                "1. Sadece verilen bağlamı kullan.\n"
                "2. Bilgi yoksa 'Bu bilgi notlarda bulunmuyor.' de.\n"
                "3. ASLA liste yapma (1. 2. 3. diye numaralandırma).\n"
                "4. Cevabı düz bir paragraf olarak yaz.\n"
                "5. Tekrara düşersen hemen sus."
            )
            
            response = chat_client.complete_chat(messages=[{"role": "system", "content": sistem_mesaji}, {"role": "user", "content": soru}])
            cevap = clean_output(response.choices[0].message.content)
            
            if "notlarda bulunmuyor" in cevap.lower() or len(cevap) < 5:
                cevap = "Bu bilgi notlarda bulunmuyor."
            
            print("\n🤖 Yapay Zeka:")
            for char in cevap:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.02) # Buradaki 0.02 saniye harf hızıdır, istersen 0.01 yapıp hızlandırabilirsin.
            print()
            
        except Exception as e:
            print(f"\n[HATA]: {e}")

if __name__ == "__main__":
    main()
    