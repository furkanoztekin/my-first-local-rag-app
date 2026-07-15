import sys
import chromadb
from foundry_local_sdk import FoundryLocalManager, Configuration

def on_progress(ep_name: str, percent: float) -> None:
    print(f"\r⚡ {ep_name} kontrol ediliyor/indiriliyor... %{percent:5.1f}", end="", flush=True)

def main():
    print("🚀 Sistem başlatılıyor (NVIDIA RTX 5070 Aktif)...")
    
    # 1. SDK Başlatma
    config = Configuration(app_name="my-first-local-rag-app")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    
    # 2. CUDA Kayıt ve Doğrulama
    eps = manager.discover_eps()
    cuda_ep = next((ep for ep in eps if ep.name == "CUDAExecutionProvider"), None)
    
    if cuda_ep and not cuda_ep.is_registered:
        print("\n⚠️ CUDA dosyaları SDK'ya kaydediliyor...")
        manager.download_and_register_eps(["CUDAExecutionProvider"], progress_callback=on_progress)
        print("\n✅ CUDA entegrasyonu tamamlandı!")
    
    # 3. Modelleri Yükleme
    catalog = manager.catalog
    
    print(" - Embedding modeli (Qwen) hazırlanıyor...")
    embed_model = catalog.get_model("qwen3-embedding-0.6b") 
    if not embed_model.is_cached: embed_model.download()
    embed_model.load()
    embed_client = embed_model.get_embedding_client()
    
    print(" - Phi-3.5 modeli (GPU üzerinden) yükleniyor...")
    chat_model = catalog.get_model("phi-3.5-mini")
    chat_model.load() 
    
    chat_client = chat_model.get_chat_client()
    
    # KESİN FREN SİSTEMİ (Anti-Loop Ayarları)
    chat_client.settings.temperature = 0.3  # Modelin saçmalamasını engeller
    chat_client.settings.top_p = 0.9        # Kelime yelpazesini daraltır
    chat_client.settings.max_tokens = 800   # 🛑 250 kelimeden sonra fiziksel olarak susar!
    
    # 4. Vektör Veritabanı Bağlantısı
    print(" - Vektör veritabanına bağlanılıyor...")
    db_client = chromadb.PersistentClient(path="./vektor_db")
    try:
        collection = db_client.get_collection(name="pdf_notlarim")
    except Exception as e:
        print(f"❌ Koleksiyon bulunamadı: {e}\nLütfen önce ingest.py çalıştırarak PDF'leri veritabanına ekle.")
        sys.exit()
    
    print("\n" + "="*60)
    print("✅ SİSTEM HAZIR! Vize notlarına soru sorabilirsin. (Çıkmak için: q)")
    print("="*60)
    
    # 5. Chat Döngüsü
    while True:
        soru = input("\nSen: ")
        if soru.lower() in ['q', 'çıkış', 'exit']: break
        
        try:
            print("🔎 Notlar taranıyor...")
            soru_vektoru = embed_client.generate_embeddings(inputs=[soru]).data[0].embedding
            sonuclar = collection.query(query_embeddings=[soru_vektoru], n_results=3)
            baglam = "\n\n--- YENİ NOT ---\n\n".join(sonuclar['documents'][0])
            
            # KATI SİSTEM MESAJI
            sistem_mesaji = f"""Sen bir akademik asistansın. Sadece aşağıdaki notlara dayanarak kısa, net ve anlaşılır bir cevap ver. 
            KURALLAR: 
            1. Asla aynı cümleyi tekrar etme.
            2. 3-4 cümleyi geçme.
            3. Cevabını bitirince doğrudan sus.
            Notlar: {baglam}"""
            
            response = chat_client.complete_chat(messages=[
                {"role": "system", "content": sistem_mesaji},
                {"role": "user", "content": soru}
            ])
            print(f"\n🤖 Yapay Zeka: {response.choices[0].message.content}")
        except Exception as e:
            print(f"\n[HATA]: {e}")

if __name__ == "__main__":
    main()