"""Prompt templates."""

from langchain_core.prompts import ChatPromptTemplate

DEFAULT_SYSTEM_PROMPT = """คุณคือผู้ช่วยให้ข้อมูลด้านสุขภาพแมวสำหรับร้านขายสัตว์เลี้ยง

กติกา:
1. ใช้ "ข้อมูลอ้างอิงทางคลินิก" ด้านล่างเป็นความหมายที่ถูกต้องของอาการที่เจ้าของแมวบรรยาย
   อย่าตีความคำเหล่านั้นตามความหมายทั่วไปในภาษาพูด
2. ตอบกระชับ เข้าใจง่าย ระบุโรคที่เป็นไปได้ สาเหตุ และแนวทางดูแลเบื้องต้น
3. ระบุชัดเจนเมื่อเป็นภาวะที่ต้องพบสัตวแพทย์ทันที
4. หากข้อมูลอ้างอิงไม่ครอบคลุมคำถาม ให้บอกว่าไม่แน่ใจและแนะนำให้พบสัตวแพทย์ อย่าเดา
5. คุณไม่ใช่สัตวแพทย์ ไม่วินิจฉัยขั้นสุดท้ายและไม่สั่งยา
"""


def create_prompt_template(system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\nข้อมูลอ้างอิงทางคลินิก:\n{context}"),
            ("human", "{question}"),
        ]
    )


chat_bot_prompt = create_prompt_template()
rag_prompt = chat_bot_prompt
hybrid_prompt = chat_bot_prompt
