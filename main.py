from foundry_local_sdk import FoundryLocalManager, Configuration

def main():
    print("1. Sistem başlatılıyor...")
    config = Configuration(app_name="my-first-local-rag-app") 
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    
    model_alias = "phi-3.5-mini"
    model = manager.catalog.get_model(model_alias)
    
    print("2. Model ekran kartına (VRAM) yükleniyor...")
    model.load()
    
    print("3. Sohbet istemcisi (ChatClient) başlatılıyor...")
    client = model.get_chat_client()
    
    print("4. Modele ilk mesaj gönderiliyor. (RTX 5070 devrede!)...\n")
    
    response = client.complete_chat(
        messages=[
            {"role": "system", "content": "Sen bilgisayarda yerel olarak çalışan, Türkçe konuşan bir yapay zeka asistanısın. Samimi, kısa ve öz cevaplar verirsin."},
            {"role": "user", "content": "Merhaba! Kendi bilgisayarımda, kendi donanımımla çalıştığını hissetmek harika. Bana kendini 2 cümle ile tanıtır mısın?"}
        ]
    )
    
    print("="*50)
    print("🤖 YAPAY ZEKANIN CEVABI:")
    print("="*50)
    print(response.choices[0].message.content)  
    print("="*50)

if __name__ == "__main__":
    main()