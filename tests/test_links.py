"""商品链接识别测试（PRD §5.3 输入层）。

核心约束：**原样返回、绝不清洗参数**。实测同一条链接带 `uin` 能反查到商品、
去掉 `uin` 即被平台拒绝——任何"规范化 URL"的习惯做法都会正好把凭据剥掉。
"""

from priceradar.agent.llm_nodes import extract_keyword
from priceradar.sources.links import extract_pdd_link

REAL_LINK = "https://mobile.yangkeduo.com/goods.html?goods_id=726516013895&uin=AQ6OH5TPS2RZ7GMOS6H2RBQTMA_GEXDA"
SHORT_LINK = "https://p.pinduoduo.com/2bfkaAO1?sc=EFAC"


# —— 识别 ——


def test_recognises_real_shared_link():
    assert extract_pdd_link(REAL_LINK) == REAL_LINK


def test_recognises_short_link():
    assert extract_pdd_link(SHORT_LINK) == SHORT_LINK


def test_extracts_link_embedded_in_natural_language():
    """用户不会只粘链接——常见是"帮我看看这个 <链接> 值不值"。"""
    text = f"帮我看看这个 {REAL_LINK} 值不值"
    assert extract_pdd_link(text) == REAL_LINK


def test_strips_trailing_chinese_punctuation_only():
    """粘贴时常带进中文标点；只清尾部标点，**不碰 URL 内部任何参数**。"""
    assert extract_pdd_link(f"看这个：{REAL_LINK}。") == REAL_LINK
    assert extract_pdd_link(f"【{REAL_LINK}】") == REAL_LINK


def test_preserves_all_query_params():
    """最关键的一条：uin 等平台参数必须原样保留（去掉就会被平台拒绝）。"""
    extracted = extract_pdd_link(f"{REAL_LINK} 谢谢")
    assert extracted is not None
    assert "uin=" in extracted
    assert "goods_id=726516013895" in extracted


def test_returns_none_for_plain_text():
    assert extract_pdd_link("QCY N20 降噪耳机") is None
    assert extract_pdd_link("") is None
    assert extract_pdd_link(None) is None


def test_other_platform_links_are_not_claimed():
    """京东/淘宝链接不归本函数管——别把不是拼多多的链接当拼多多的用。"""
    assert extract_pdd_link("https://item.jd.com/100012043978.html") is None
    assert extract_pdd_link("https://item.taobao.com/item.htm?id=123") is None


# —— 与查询理解的集成（短路优先于 LLM）——


def test_extract_keyword_short_circuits_link_before_llm():
    """链接必须短路——否则会被当成自然语言交给 LLM 提取关键词，URL 直接被毁掉。"""
    keyword, method = extract_keyword(REAL_LINK, None)
    assert method == "link"
    assert keyword == REAL_LINK


def test_extract_keyword_unchanged_for_normal_queries():
    _, method = extract_keyword("帮我看看QCY N20哪里买划算", None)
    assert method == "rule"      # 无 LLM 客户端时走规则兜底，不受影响
