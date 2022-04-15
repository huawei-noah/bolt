<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:output encoding="UTF-8" indent="yes" method="xml"></xsl:output>
<xsl:template match="/">
  <checkstyle>
    <xsl:attribute name="version">
      <xsl:value-of select="results/cppcheck[@version]"/>
    </xsl:attribute>
    <xsl:for-each-group select="results/errors/error" group-by="(location/@file)[1]">
      <file>
        <xsl:attribute name="name">
          <xsl:value-of select="current-grouping-key()"/>
        </xsl:attribute>
        <xsl:for-each select="current-group()">
          <error line="{(location/@line)[1]}" message="{@msg}" severity="{@severity}" source="{@id}" />
        </xsl:for-each>
      </file>
    </xsl:for-each-group>
  </checkstyle>
</xsl:template>
</xsl:stylesheet>